#!/usr/bin/env bash

distil3d_run_usage() {
  cat <<'EOF'
Usage:
  ./distil3d run --profile <profile-directory> <job-directory> [--run-id <id>] [--detach]
EOF
}

distil3d_resume_usage() {
  cat <<'EOF'
Usage:
  ./distil3d resume --profile <profile-directory> <job-name>/<run-id>/<source-attempt-id> --checkpoint <container-path> [--detach]
EOF
}

distil3d_stop_usage() {
  cat <<'EOF'
Usage:
  ./distil3d stop --profile <profile-directory> <job-name>/<run-id>[/<attempt-id>] [--grace-seconds <n>]
EOF
}

distil3d_new_id() {
  local timestamp
  local suffix

  timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  suffix="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
  printf '%s-%s\n' "${timestamp}" "${suffix}"
}

distil3d_attempt_mount_arguments() {
  DISTIL3D_ATTEMPT_MOUNTS=(
    --mount "type=bind,src=${DISTIL3D_DATA_ROOT},dst=/data,readonly"
    --mount "type=bind,src=${DISTIL3D_MODELS_ROOT},dst=/models,readonly"
    --mount "type=bind,src=${DISTIL3D_PREPARED_ROOT},dst=/prepared"
    --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs"
    --mount "type=bind,src=${DISTIL3D_CACHE_ROOT},dst=/cache"
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/jobs,dst=/project-config/jobs,readonly"
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/recipes,dst=/project-config/recipes,readonly"
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/selections,dst=/project-config/selections,readonly"
  )
}

distil3d_create_attempt() {
  local attempt_kind="$1"
  local job_name="$2"
  local run_id="$3"
  local job_file="$4"
  local resume_record="$5"
  local selected_checkpoint="$6"
  local attempt_id
  local attempt_ref
  local attempt_root
  local container_name
  local container_id
  local -a execute_source
  local -a checkpoint_argument=()

  attempt_id="$(distil3d_new_id)"
  attempt_ref="${job_name}/${run_id}/${attempt_id}"
  attempt_root="${DISTIL3D_RUNS_ROOT}/${job_name}/${run_id}/attempts/${attempt_id}"
  container_name="distil3d-${attempt_id}"
  mkdir -p -- "${attempt_root}"

  DISTIL3D_ATTEMPT_ID="${attempt_id}"
  DISTIL3D_ATTEMPT_REF="${attempt_ref}"
  DISTIL3D_ATTEMPT_CONTAINER_ID=""
  DISTIL3D_ATTEMPT_CONTAINER_NAME="${container_name}"
  DISTIL3D_ATTEMPT_CONTAINER_STATE="creating"
  printf 'attempt=%s\n' "${attempt_ref}"

  if [[ "${attempt_kind}" == "run" ]]; then
    execute_source=(--job "${job_file}")
  else
    execute_source=(--resume-record "${resume_record}")
    checkpoint_argument=(--selected-checkpoint "${selected_checkpoint}")
  fi

  distil3d_attempt_mount_arguments
  container_id="$(
    docker create \
      --name "${container_name}" \
      --label "io.distil3d.job=${job_name}" \
      --label "io.distil3d.run-id=${run_id}" \
      --label "io.distil3d.attempt-id=${attempt_id}" \
      --label "io.distil3d.attempt-ref=${attempt_ref}" \
      --user "$(id -u):$(id -g)" \
      "${DISTIL3D_RESOURCE_ARGUMENTS[@]}" \
      "${DISTIL3D_ATTEMPT_MOUNTS[@]}" \
      "${DISTIL3D_SELECTED_IMAGE_ID}" \
      /opt/envs/core/bin/python \
      -m novel_view.cli.main \
      execute \
      "${execute_source[@]}" \
      --config-root /project-config \
      --run-id "${run_id}" \
      --attempt-id "${attempt_id}" \
      --attempt-kind "${attempt_kind}" \
      --attempt-root "/runs/${job_name}/${run_id}/attempts/${attempt_id}" \
      --container-name "${container_name}" \
      --image-variant "${DISTIL3D_SELECTED_IMAGE_VARIANT}" \
      --image-id "${DISTIL3D_SELECTED_IMAGE_ID}" \
      --image-source-revision "${DISTIL3D_SELECTED_IMAGE_SOURCE_REVISION}" \
      --image-source-dirty "${DISTIL3D_SELECTED_IMAGE_SOURCE_DIRTY}" \
      --image-lock-revision "${DISTIL3D_SELECTED_IMAGE_LOCK_REVISION}" \
      "${checkpoint_argument[@]}"
  )" || return $?

  DISTIL3D_ATTEMPT_CONTAINER_ID="${container_id}"
  DISTIL3D_ATTEMPT_CONTAINER_STATE="created"
}

distil3d_follow_attempt() {
  local profile_argument="$1"
  local follower_pid=""
  local interrupted=0
  local container_exit_code

  trap '
    interrupted=1
    if [[ -n "${follower_pid}" ]]; then
      kill -TERM "${follower_pid}" 2>/dev/null || true
    fi
  ' INT

  docker logs --follow "${DISTIL3D_ATTEMPT_CONTAINER_ID}" &
  follower_pid=$!
  if wait "${follower_pid}"; then
    :
  else
    :
  fi
  trap - INT

  if [[ ${interrupted} -eq 1 ]]; then
    printf 'log following interrupted: attempt=%s\n' "${DISTIL3D_ATTEMPT_REF}"
    printf 'recover: ./distil3d status --profile %q %q\n' \
      "${profile_argument}" "${DISTIL3D_ATTEMPT_REF}"
    printf 'recover: ./distil3d logs --profile %q %q -f\n' \
      "${profile_argument}" "${DISTIL3D_ATTEMPT_REF}"
    return 130
  fi

  distil3d_inspect_container "${DISTIL3D_ATTEMPT_CONTAINER_ID}" || return $?
  case "${DISTIL3D_CONTAINER_STATE}" in
    exited | dead)
      container_exit_code="${DISTIL3D_CONTAINER_EXIT_CODE}"
      return "${container_exit_code}"
      ;;
    *)
      printf 'log stream ended before terminal container state: attempt=%s state=%s\n' \
        "${DISTIL3D_ATTEMPT_REF}" "${DISTIL3D_CONTAINER_STATE}" >&2
      printf 'recover: ./distil3d status --profile %q %q\n' \
        "${profile_argument}" "${DISTIL3D_ATTEMPT_REF}" >&2
      return 1
      ;;
  esac
}

distil3d_start_attempt() {
  local profile_argument="$1"
  local detach="$2"

  docker start "${DISTIL3D_ATTEMPT_CONTAINER_ID}" >/dev/null
  DISTIL3D_ATTEMPT_CONTAINER_STATE="running"
  if [[ "${detach}" == "true" ]]; then
    return 0
  fi
  distil3d_follow_attempt "${profile_argument}"
}

distil3d_run_main() {
  local profile_argument=""
  local job_argument=""
  local run_id=""
  local detach=false
  local job_file

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
      --run-id)
        if [[ $# -lt 2 ]]; then
          printf 'missing value for --run-id\n' >&2
          return 2
        fi
        run_id="$2"
        shift 2
        ;;
      --detach)
        detach=true
        shift
        ;;
      -h | --help)
        distil3d_run_usage
        return 0
        ;;
      --*)
        printf 'unknown run argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${job_argument}" ]]; then
          printf 'run accepts exactly one job directory\n' >&2
          return 2
        fi
        job_argument="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" ]]; then
    printf 'run requires --profile\n' >&2
    return 2
  fi
  if [[ -z "${job_argument}" ]]; then
    printf 'run requires a job directory\n' >&2
    return 2
  fi
  if [[ -z "${run_id}" ]]; then
    run_id="$(distil3d_new_id)"
  fi

  distil3d_load_profile "${profile_argument}"
  job_file="$(distil3d_job_container_file "${job_argument}")"
  distil3d_select_job_image "${job_file}"
  distil3d_final_job_plan "${job_file}" >/dev/null
  distil3d_create_attempt \
    run \
    "${DISTIL3D_SELECTED_JOB_NAME}" \
    "${run_id}" \
    "${job_file}" \
    "" \
    ""
  distil3d_start_attempt "${profile_argument}" "${detach}"
}

distil3d_read_resume_record() {
  local job_name="$1"
  local run_id="$2"
  local source_attempt_id="$3"
  local selection_output

  selection_output="$(
    docker run \
      --rm \
      --network none \
      --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs,readonly" \
      --entrypoint "" \
      "${DISTIL3D_CORE_IMAGE}" \
      /opt/envs/core/bin/python \
      -m novel_view.cli.main \
      select-resume \
      --runs-root /runs \
      --job-name "${job_name}" \
      --run-id "${run_id}" \
      --attempt-id "${source_attempt_id}"
  )" || return $?
  distil3d_read_job_selection "${selection_output}" true
  if [[ "${DISTIL3D_SELECTED_JOB_NAME}" != "${job_name}" ]]; then
    printf 'resume source belongs to job %s, not %s\n' \
      "${DISTIL3D_SELECTED_JOB_NAME}" "${job_name}" >&2
    return 2
  fi
}

distil3d_resume_main() {
  local profile_argument=""
  local run_ref=""
  local selected_checkpoint=""
  local detach=false
  local job_name
  local run_id
  local source_attempt_id

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
      --checkpoint)
        if [[ $# -lt 2 ]]; then
          printf 'missing value for --checkpoint\n' >&2
          return 2
        fi
        selected_checkpoint="$2"
        shift 2
        ;;
      --detach)
        detach=true
        shift
        ;;
      -h | --help)
        distil3d_resume_usage
        return 0
        ;;
      --*)
        printf 'unknown resume argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${run_ref}" ]]; then
          printf 'resume accepts exactly one run reference\n' >&2
          return 2
        fi
        run_ref="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" ]]; then
    printf 'resume requires --profile\n' >&2
    return 2
  fi
  if [[ -z "${run_ref}" ]]; then
    printf 'resume requires <job-name>/<run-id>/<source-attempt-id>\n' >&2
    return 2
  fi
  if [[ -z "${selected_checkpoint}" ]]; then
    printf 'resume requires --checkpoint\n' >&2
    return 2
  fi

  distil3d_parse_attempt_reference "${run_ref}"
  if [[ -z "${DISTIL3D_REFERENCE_ATTEMPT_ID}" ]]; then
    printf 'resume requires <job-name>/<run-id>/<source-attempt-id>\n' >&2
    return 2
  fi
  job_name="${DISTIL3D_REFERENCE_JOB_NAME}"
  run_id="${DISTIL3D_REFERENCE_RUN_ID}"
  source_attempt_id="${DISTIL3D_REFERENCE_ATTEMPT_ID}"
  distil3d_load_profile "${profile_argument}"
  distil3d_read_resume_record "${job_name}" "${run_id}" "${source_attempt_id}"
  distil3d_inspect_selected_image
  distil3d_final_job_plan "" "${DISTIL3D_SELECTED_RESUME_RECORD}" \
    "${selected_checkpoint}" >/dev/null

  distil3d_create_attempt \
    resume \
    "${job_name}" \
    "${run_id}" \
    "" \
    "${DISTIL3D_SELECTED_RESUME_RECORD}" \
    "${selected_checkpoint}"
  distil3d_start_attempt "${profile_argument}" "${detach}"
}

distil3d_stop_main() {
  local profile_argument=""
  local reference=""
  local grace_seconds="10"

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
      --grace-seconds)
        if [[ $# -lt 2 ]]; then
          printf 'missing value for --grace-seconds\n' >&2
          return 2
        fi
        grace_seconds="$2"
        shift 2
        ;;
      -h | --help)
        distil3d_stop_usage
        return 0
        ;;
      --*)
        printf 'unknown stop argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${reference}" ]]; then
          printf 'stop accepts exactly one attempt reference\n' >&2
          return 2
        fi
        reference="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" || -z "${reference}" ]]; then
    distil3d_stop_usage >&2
    return 2
  fi

  distil3d_load_profile "${profile_argument}"
  distil3d_resolve_attempt "${reference}"
  if [[ -z "${DISTIL3D_RESOLVED_CONTAINER_ID}" ]]; then
    printf 'container is absent: %s\n' "${DISTIL3D_RESOLVED_ATTEMPT_REF}" >&2
    return 1
  fi

  distil3d_inspect_resolved_container
  case "${DISTIL3D_CONTAINER_STATE}" in
    running | paused | restarting)
      docker stop \
        --signal SIGTERM \
        --time "${grace_seconds}" \
        "${DISTIL3D_RESOLVED_CONTAINER_ID}" >/dev/null || return $?
      ;;
    exited | dead)
      printf 'container already finished: %s\n' \
        "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
      ;;
    *)
      printf 'container is not running: %s state=%s\n' \
        "${DISTIL3D_RESOLVED_ATTEMPT_REF}" \
        "${DISTIL3D_CONTAINER_STATE}" >&2
      return 1
      ;;
  esac

  distil3d_inspect_resolved_container || return $?
  distil3d_print_container_state
  printf 'attempt_ref=%s\n' "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
  case "${DISTIL3D_CONTAINER_STATE}" in
    exited | dead)
      return 0
      ;;
    *)
      printf 'container did not reach a terminal state: %s state=%s\n' \
        "${DISTIL3D_RESOLVED_ATTEMPT_REF}" \
        "${DISTIL3D_CONTAINER_STATE}" >&2
      return 1
      ;;
  esac
}
