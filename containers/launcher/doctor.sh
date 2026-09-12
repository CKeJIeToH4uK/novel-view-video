#!/usr/bin/env bash

distil3d_doctor_usage() {
  cat <<'EOF'
Usage:
  ./distil3d doctor host --profile <profile-directory>
  ./distil3d doctor job --profile <profile-directory> <job-directory>
  ./distil3d doctor waymo-ingress --profile <profile-directory> -- --waymo-root /data/<dataset> --split-config /project-config/selections/<split.yaml> --max-rss-mib 1024
  ./distil3d doctor waymo-raster --profile <profile-directory> -- --waymo-root /data/<dataset> --split-config /project-config/selections/<split.yaml>
EOF
}

distil3d_doctor_probe_root() {
  local root_name="$1"
  local root_path="$2"
  local probe_file
  local write_status

  probe_file="$(
    CDPATH= cd -- "${root_path}" &&
      mktemp "${PWD%/}/.distil3d-doctor.XXXXXX"
  )"
  if printf 'distil3d doctor host\n' > "${probe_file}"; then
    :
  else
    write_status=$?
    rm -f -- "${probe_file}"
    return "${write_status}"
  fi
  rm -- "${probe_file}"
  printf 'profile.%s_writable=ok\n' "${root_name}"
}

distil3d_doctor_host() {
  local docker_version
  local docker_architecture
  local docker_storage_driver
  local docker_root_directory
  local docker_available_kib

  if ! command -v docker >/dev/null 2>&1; then
    printf 'docker CLI not found\n' >&2
    return 1
  fi

  docker_version="$(docker version --format '{{.Server.Version}}')"
  docker_architecture="$(docker info --format '{{.Architecture}}')"
  docker_storage_driver="$(docker info --format '{{.Driver}}')"
  docker_root_directory="$(docker info --format '{{.DockerRootDir}}')"
  docker_available_kib="$(
    LC_ALL=C df -Pk "${docker_root_directory}" | awk 'NR == 2 { print $4 }'
  )"

  printf 'docker.version=%s\n' "${docker_version}"
  printf 'docker.architecture=%s\n' "${docker_architecture}"
  printf 'docker.storage_driver=%s\n' "${docker_storage_driver}"
  printf 'docker.root_directory=%s\n' "${docker_root_directory}"
  printf 'docker.available_kib=%s\n' "${docker_available_kib}"
  printf 'profile.directory=%s\n' "${DISTIL3D_PROFILE_DIRECTORY}"
  printf 'profile.data_root=%s\n' "${DISTIL3D_DATA_ROOT}"
  printf 'profile.models_root=%s\n' "${DISTIL3D_MODELS_ROOT}"
  printf 'profile.prepared_root=%s\n' "${DISTIL3D_PREPARED_ROOT}"
  printf 'profile.runs_root=%s\n' "${DISTIL3D_RUNS_ROOT}"
  printf 'profile.cache_root=%s\n' "${DISTIL3D_CACHE_ROOT}"
  printf 'profile.gpu_ids=%s\n' "${DISTIL3D_GPU_IDS}"

  distil3d_doctor_probe_root prepared "${DISTIL3D_PREPARED_ROOT}"
  distil3d_doctor_probe_root runs "${DISTIL3D_RUNS_ROOT}"
  distil3d_doctor_probe_root cache "${DISTIL3D_CACHE_ROOT}"
}

distil3d_doctor_job() {
  local job_file="$1"

  distil3d_select_job_image "${job_file}" || return $?
  distil3d_final_job_plan "${job_file}" >/dev/null || return $?
  distil3d_attempt_mount_arguments
  docker run \
    --rm \
    --user "$(id -u):$(id -g)" \
    "${DISTIL3D_RESOURCE_ARGUMENTS[@]}" \
    "${DISTIL3D_ATTEMPT_MOUNTS[@]}" \
    --entrypoint "" \
    "${DISTIL3D_SELECTED_IMAGE_ID}" \
    /opt/envs/core/bin/python \
    -m novel_view.cli.main \
    doctor \
    --job "${job_file}" \
    --config-root /project-config
}

distil3d_doctor_waymo() {
  local diagnostic="$1"
  local image_id
  shift

  image_id="$(docker image inspect --format '{{.Id}}' "${DISTIL3D_CORE_IMAGE}")" || return $?
  distil3d_select_gpu_resources 0
  distil3d_attempt_mount_arguments
  docker run --rm --user "$(id -u):$(id -g)" \
    "${DISTIL3D_RESOURCE_ARGUMENTS[@]}" \
    "${DISTIL3D_ATTEMPT_MOUNTS[@]}" \
    --entrypoint "" "${image_id}" \
    /opt/envs/core/bin/python -m novel_view.cli.main \
    doctor-waymo "${diagnostic}" "$@"
}

distil3d_doctor_main() {
  local doctor_command=""
  local profile_argument=""
  local job_argument=""
  local job_file
  local -a diagnostic_arguments=()

  if [[ $# -eq 0 ]]; then
    distil3d_doctor_usage >&2
    return 2
  fi

  case "$1" in
    host)
      doctor_command=host
      shift
      ;;
    job)
      doctor_command=job
      shift
      ;;
    waymo-ingress | waymo-raster)
      doctor_command="$1"
      shift
      ;;
    -h | --help | help)
      distil3d_doctor_usage
      return 0
      ;;
    *)
      printf 'unknown doctor command: %s\n' "$1" >&2
      distil3d_doctor_usage >&2
      return 2
      ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        if [[ $# -lt 2 ]]; then
          printf 'missing value for --profile\n' >&2
          return 2
        fi
        profile_argument="$2"
        shift 2
        ;;
      -h | --help)
        distil3d_doctor_usage
        return 0
        ;;
      --)
        if [[ "${doctor_command}" != waymo-* ]]; then
          printf 'doctor %s does not accept diagnostic arguments\n' "${doctor_command}" >&2
          return 2
        fi
        shift
        diagnostic_arguments=("$@")
        break
        ;;
      --*)
        printf 'unknown doctor %s argument: %s\n' "${doctor_command}" "$1" >&2
        return 2
        ;;
      *)
        if [[ "${doctor_command}" != "job" || -n "${job_argument}" ]]; then
          printf 'doctor %s accepts no additional argument: %s\n' \
            "${doctor_command}" "$1" >&2
          return 2
        fi
        job_argument="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" ]]; then
    printf 'doctor %s requires --profile\n' "${doctor_command}" >&2
    return 2
  fi
  if [[ "${doctor_command}" == "job" && -z "${job_argument}" ]]; then
    printf 'doctor job requires a job directory\n' >&2
    return 2
  fi

  distil3d_load_profile "${profile_argument}"
  if [[ "${doctor_command}" == "host" ]]; then
    distil3d_doctor_host
    return
  fi
  if [[ "${doctor_command}" == waymo-* ]]; then
    distil3d_doctor_waymo "${doctor_command#waymo-}" "${diagnostic_arguments[@]}"
    return
  fi
  job_file="$(distil3d_job_container_file "${job_argument}")"
  distil3d_doctor_job "${job_file}"
}
