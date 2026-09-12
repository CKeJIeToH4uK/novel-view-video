#!/usr/bin/env bash

readonly DISTIL3D_CPU_TEST_IMAGE="distil3d:cpu-test"
readonly DISTIL3D_CORE_IMAGE="distil3d:cuda124-sm80-core"
readonly DISTIL3D_MOGE_IMAGE="distil3d:cuda124-sm80-moge"

distil3d_production_lock_inventory() {
  local relative_path

  for relative_path in \
    containers/diffusion/requirements/core.conda.lock \
    containers/diffusion/requirements/core.pip.lock \
    containers/diffusion/requirements/gen3c-eval.conda.lock \
    containers/diffusion/requirements/gen3c-eval.pip.lock \
    containers/diffusion/requirements/vggt.conda.lock \
    containers/diffusion/requirements/vggt.pip.lock \
    containers/diffusion/requirements/moge-gen3c.pip.lock
  do
    printf '%s\tsha256:%s\n' "${relative_path}" \
      "$(distil3d_sha256_file "${DISTIL3D_REPOSITORY_ROOT}/${relative_path}")"
  done
}

distil3d_require_candidate_checkout() {
  local source_revision="$1"

  if [[ "$(git -C "${DISTIL3D_REPOSITORY_ROOT}" rev-parse HEAD)" != "${source_revision}" || \
    "$(distil3d_source_dirty)" != false ]]
  then
    printf 'candidate export requires an unchanged source revision and no dirty files\n' >&2
    return 2
  fi
}

distil3d_check_candidate_image() {
  local image_id="$1"
  local variant="$2"
  local source_revision="$3"
  local lock_revision="$4"
  local actual
  local expected

  actual="$(docker image inspect --format \
    '{{.Os}}/{{.Architecture}}|{{index .Config.Labels "org.opencontainers.image.revision"}}|{{index .Config.Labels "io.distil3d.source.dirty"}}|{{index .Config.Labels "io.distil3d.lock.revision"}}|{{index .Config.Labels "io.distil3d.variant"}}|{{index .Config.Labels "io.distil3d.gen3c.revision"}}|{{index .Config.Labels "io.distil3d.dinov2.revision"}}' \
    "${image_id}")" || return $?
  expected="linux/amd64|${source_revision}|false|${lock_revision}|${variant}"
  expected+='|db2ffe12ced12ddafcec5e0422ee46ce8520746b|7764ea0f912e53c92e82eb78a2a1631e92725fc8'
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'image metadata does not match the candidate: %s\n' "${variant}" >&2
    return 2
  fi
}

distil3d_build_usage() {
  cat <<'EOF'
Usage:
  ./distil3d build [--variant core|moge] [--candidate-output <new-directory>]
EOF
}

distil3d_build_main() {
  local selected_variant="core"
  local candidate_output=""
  local output_parent
  local source_revision
  local source_dirty
  local inventory
  local lock_revision
  local variant
  local image_tag
  local image_id
  local core_id=""
  local moge_id=""
  local prefix
  local -a variants=(core)

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --variant | --candidate-output)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          printf 'missing value for %s\n' "$1" >&2
          return 2
        fi
        if [[ "$1" == --variant ]]; then
          selected_variant="$2"
        else
          candidate_output="$2"
        fi
        shift 2
        ;;
      -h | --help)
        distil3d_build_usage
        return 0
        ;;
      *)
        printf 'unknown build argument: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done

  case "${selected_variant}" in
    core) ;;
    moge) variants+=(moge) ;;
    *) printf 'unsupported build variant: %s\n' "${selected_variant}" >&2; return 2 ;;
  esac
  if [[ -n "${candidate_output}" && "${selected_variant}" != moge ]]; then
    printf 'candidate export requires --variant moge\n' >&2
    return 2
  fi

  source_revision="$(git -C "${DISTIL3D_REPOSITORY_ROOT}" rev-parse HEAD)"
  source_dirty="$(distil3d_source_dirty)"
  inventory="$(distil3d_production_lock_inventory)"
  lock_revision="sha256:$(printf '%s\n' "${inventory}" | distil3d_sha256_stream)"

  if [[ -n "${candidate_output}" ]]; then
    distil3d_require_candidate_checkout "${source_revision}"
    output_parent="$(CDPATH= cd -- "$(dirname -- "${candidate_output}")" && pwd -P)"
    candidate_output="${output_parent}/$(basename -- "${candidate_output}")"
    case "${candidate_output}" in
      "${DISTIL3D_REPOSITORY_ROOT}" | "${DISTIL3D_REPOSITORY_ROOT}/"*)
        printf 'candidate output must be outside the checkout\n' >&2
        return 2
        ;;
    esac
    mkdir -m 755 -- "${candidate_output}"
    distil3d_test_main all
  fi

  for variant in "${variants[@]}"; do
    if [[ -n "${candidate_output}" ]]; then
      distil3d_require_candidate_checkout "${source_revision}"
    fi
    image_tag="${DISTIL3D_CORE_IMAGE}"
    if [[ "${variant}" == moge ]]; then image_tag="${DISTIL3D_MOGE_IMAGE}"; fi
    docker build \
      --platform linux/amd64 \
      --file "${DISTIL3D_REPOSITORY_ROOT}/containers/diffusion/Dockerfile" \
      --target "production-${variant}" \
      --tag "${image_tag}" \
      --build-arg "DISTIL3D_SOURCE_REVISION=${source_revision}" \
      --build-arg "DISTIL3D_SOURCE_DIRTY=${source_dirty}" \
      --build-arg "DISTIL3D_LOCK_REVISION=${lock_revision}" \
      "${DISTIL3D_REPOSITORY_ROOT}"
    image_id="$(docker image inspect --format '{{.Id}}' "${image_tag}")"
    if [[ "${variant}" == core ]]; then core_id="${image_id}"; else moge_id="${image_id}"; fi
    printf 'built %s=%s\n' "${variant}" "${image_id}"
    if [[ -n "${candidate_output}" ]]; then
      distil3d_require_candidate_checkout "${source_revision}"
      distil3d_check_candidate_image "${image_id}" "${variant}" "${source_revision}" "${lock_revision}"
      for prefix in core gen3c vggt; do
        docker run --rm --network none --entrypoint "" "${image_id}" \
          "/opt/envs/${prefix}/bin/python" -m novel_view.diagnostics.image \
          --prefix "${prefix}" --variant "${variant}"
      done
    fi
  done

  if [[ -z "${candidate_output}" ]]; then return 0; fi
  distil3d_require_candidate_checkout "${source_revision}"
  if [[ "$(docker image inspect --format '{{.Id}}' "${DISTIL3D_CORE_IMAGE}")" != "${core_id}" || \
    "$(docker image inspect --format '{{.Id}}' "${DISTIL3D_MOGE_IMAGE}")" != "${moge_id}" ]]
  then
    printf 'candidate image tags changed during the build\n' >&2
    return 2
  fi
  docker save --output "${candidate_output}/images.tar" \
    "${DISTIL3D_CORE_IMAGE}" "${DISTIL3D_MOGE_IMAGE}"
  printf '%s\n' "${inventory}" | docker run --rm -i --network none \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=${candidate_output},dst=/candidate-out" \
    --entrypoint "" "${core_id}" /opt/envs/core/bin/python \
    -m novel_view.cli.candidate write \
    --source-revision "${source_revision}" --lock-revision "${lock_revision}" \
    --core-image-id "${core_id}" --moge-image-id "${moge_id}" \
    --output /candidate-out/candidate.json
  docker run --rm --network none \
    --mount "type=bind,src=${candidate_output}/candidate.json,dst=/candidate.json,readonly" \
    --entrypoint "" "${core_id}" /opt/envs/core/bin/python \
    -m novel_view.cli.candidate read --candidate /candidate.json
  distil3d_require_candidate_checkout "${source_revision}"
  printf 'candidate=%s/candidate.json\n' "${candidate_output}"
}

distil3d_sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    shasum -a 256 | awk '{print $1}'
  fi
}

distil3d_sha256_file() {
  local file_path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file_path}" | awk '{print $1}'
  else
    shasum -a 256 "${file_path}" | awk '{print $1}'
  fi
}

distil3d_cpu_test_lock_revision() {
  local relative_path

  for relative_path in \
    containers/diffusion/requirements/core.conda.lock \
    containers/diffusion/requirements/core.pip.lock \
    containers/diffusion/requirements/cpu-test.pip.lock
  do
    printf '%s\t%s\n' \
      "${relative_path}" \
      "$(distil3d_sha256_file "${DISTIL3D_REPOSITORY_ROOT}/${relative_path}")"
  done | distil3d_sha256_stream
}

distil3d_source_dirty() {
  if [[ -n "$(git -C "${DISTIL3D_REPOSITORY_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

distil3d_build_cpu_test() {
  local source_revision
  local source_dirty
  local lock_revision

  source_revision="$(git -C "${DISTIL3D_REPOSITORY_ROOT}" rev-parse HEAD)"
  source_dirty="$(distil3d_source_dirty)"
  lock_revision="sha256:$(distil3d_cpu_test_lock_revision)"

  docker build \
    --platform linux/amd64 \
    --file "${DISTIL3D_REPOSITORY_ROOT}/containers/diffusion/Dockerfile" \
    --target cpu-test \
    --tag "${DISTIL3D_CPU_TEST_IMAGE}" \
    --build-arg "DISTIL3D_SOURCE_REVISION=${source_revision}" \
    --build-arg "DISTIL3D_SOURCE_DIRTY=${source_dirty}" \
    --build-arg "DISTIL3D_LOCK_REVISION=${lock_revision}" \
    --build-arg "DISTIL3D_IMAGE_VARIANT=cpu-test" \
    "${DISTIL3D_REPOSITORY_ROOT}"
}

distil3d_run_cpu_test() {
  docker run \
    --rm \
    --network none \
    --hostname localhost \
    "${DISTIL3D_CPU_TEST_IMAGE}" \
    "$@"
}

distil3d_run_with_config() {
  local image="$1"
  shift

  docker run \
    --rm \
    --network none \
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/jobs,dst=/project-config/jobs,readonly" \
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/recipes,dst=/project-config/recipes,readonly" \
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/selections,dst=/project-config/selections,readonly" \
    --entrypoint "" \
    "${image}" \
    "$@"
}

distil3d_inspect_selected_image() {
  case "${DISTIL3D_SELECTED_IMAGE_VARIANT}" in
    core) DISTIL3D_SELECTED_IMAGE_TAG="${DISTIL3D_CORE_IMAGE}" ;;
    moge) DISTIL3D_SELECTED_IMAGE_TAG="${DISTIL3D_MOGE_IMAGE}" ;;
    cpu-test) DISTIL3D_SELECTED_IMAGE_TAG="${DISTIL3D_CPU_TEST_IMAGE}" ;;
  esac

  DISTIL3D_SELECTED_IMAGE_ID="$(
    docker image inspect --format '{{.Id}}' "${DISTIL3D_SELECTED_IMAGE_TAG}"
  )" || return $?
  DISTIL3D_SELECTED_IMAGE_SOURCE_REVISION="$(
    docker image inspect \
      --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
      "${DISTIL3D_SELECTED_IMAGE_ID}"
  )" || return $?
  DISTIL3D_SELECTED_IMAGE_SOURCE_DIRTY="$(
    docker image inspect \
      --format '{{index .Config.Labels "io.distil3d.source.dirty"}}' \
      "${DISTIL3D_SELECTED_IMAGE_ID}"
  )" || return $?
  DISTIL3D_SELECTED_IMAGE_LOCK_REVISION="$(
    docker image inspect \
      --format '{{index .Config.Labels "io.distil3d.lock.revision"}}' \
      "${DISTIL3D_SELECTED_IMAGE_ID}"
  )" || return $?
  DISTIL3D_SELECTED_IMAGE_LABEL_VARIANT="$(
    docker image inspect \
      --format '{{index .Config.Labels "io.distil3d.variant"}}' \
      "${DISTIL3D_SELECTED_IMAGE_ID}"
  )" || return $?
  if [[ "${DISTIL3D_SELECTED_IMAGE_LABEL_VARIANT}" != \
    "${DISTIL3D_SELECTED_IMAGE_VARIANT}" ]]
  then
    printf 'selected image label does not match workflow selection\n' >&2
    return 2
  fi
}

distil3d_job_container_file() {
  local job_argument="${1%/}"
  local job_relative

  case "${job_argument}" in
    "${DISTIL3D_REPOSITORY_ROOT}/jobs/"*)
      job_relative="${job_argument#"${DISTIL3D_REPOSITORY_ROOT}/jobs/"}"
      ;;
    jobs/*)
      job_relative="${job_argument#jobs/}"
      ;;
    *)
      job_relative="${job_argument}"
      ;;
  esac
  printf '/project-config/jobs/%s/run.yaml\n' "${job_relative}"
}

distil3d_read_job_selection() {
  local selection_output="$1"
  local allow_resume_record="${2:-false}"
  local selection_line
  local selection_key
  local selection_value

  DISTIL3D_SELECTED_JOB_NAME=""
  DISTIL3D_SELECTED_IMAGE_VARIANT=""
  DISTIL3D_SELECTED_PRESET=""
  DISTIL3D_SELECTED_RESUME_RECORD=""

  while IFS= read -r selection_line || [[ -n "${selection_line}" ]]; do
    selection_key="${selection_line%%=*}"
    selection_value="${selection_line#*=}"
    case "${selection_key}" in
      job_name)
        DISTIL3D_SELECTED_JOB_NAME="${selection_value}"
        ;;
      image_variant)
        DISTIL3D_SELECTED_IMAGE_VARIANT="${selection_value}"
        ;;
      preset)
        DISTIL3D_SELECTED_PRESET="${selection_value}"
        ;;
      resume_record)
        if [[ "${allow_resume_record}" != "true" ]]; then
          printf 'unexpected resume_record in job selector\n' >&2
          return 2
        fi
        DISTIL3D_SELECTED_RESUME_RECORD="${selection_value}"
        ;;
      *)
        printf 'unknown selector field: %s\n' "${selection_key}" >&2
        return 2
        ;;
    esac
  done <<< "${selection_output}"

  if [[ -z "${DISTIL3D_SELECTED_JOB_NAME}" ]]; then
    printf 'selector did not return job_name\n' >&2
    return 2
  fi
  case "${DISTIL3D_SELECTED_IMAGE_VARIANT}" in
    core | moge | cpu-test) ;;
    *)
      printf 'unsupported image variant: %s\n' \
        "${DISTIL3D_SELECTED_IMAGE_VARIANT}" >&2
      return 2
      ;;
  esac
  case "${DISTIL3D_SELECTED_PRESET}" in
    cpu_test) DISTIL3D_SELECTED_GPU_COUNT=0 ;;
    inference_cp1) DISTIL3D_SELECTED_GPU_COUNT=1 ;;
    inference_cp2) DISTIL3D_SELECTED_GPU_COUNT=2 ;;
    training_cp4) DISTIL3D_SELECTED_GPU_COUNT=4 ;;
    *)
      printf 'unsupported execution preset: %s\n' \
        "${DISTIL3D_SELECTED_PRESET}" >&2
      return 2
      ;;
  esac
  if [[ "${allow_resume_record}" == "true" && \
    -z "${DISTIL3D_SELECTED_RESUME_RECORD}" ]]
  then
    printf 'resume selector did not return resume_record\n' >&2
    return 2
  fi
}

distil3d_select_job_image() {
  local job_file="$1"
  local selection_output

  selection_output="$(
    distil3d_run_with_config "${DISTIL3D_CORE_IMAGE}" \
      /opt/envs/core/bin/python \
      -m novel_view.cli.main \
      select \
      --job "${job_file}" \
      --config-root /project-config
  )" || return $?
  distil3d_read_job_selection "${selection_output}" || return $?
  distil3d_inspect_selected_image
}

distil3d_plan_usage() {
  cat <<'EOF'
Usage:
  ./distil3d plan --profile <profile-directory> <job-directory>
EOF
}

distil3d_plan_main() {
  local profile_argument=""
  local job_argument=""
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
      -h | --help)
        distil3d_plan_usage
        return 0
        ;;
      --*)
        printf 'unknown plan argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${job_argument}" ]]; then
          printf 'plan accepts exactly one job directory\n' >&2
          return 2
        fi
        job_argument="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" ]]; then
    printf 'plan requires --profile\n' >&2
    return 2
  fi
  if [[ -z "${job_argument}" ]]; then
    printf 'plan requires a job directory\n' >&2
    return 2
  fi

  distil3d_load_profile "${profile_argument}"
  job_file="$(distil3d_job_container_file "${job_argument}")"
  distil3d_select_job_image "${job_file}"
  distil3d_final_job_plan "${job_file}"
}

distil3d_final_job_plan() {
  local job_file="$1"
  local resume_record="${2:-}"
  local selected_checkpoint="${3:-}"
  local plan_output
  local current_image_id
  local -a source_arguments
  local -a control_command

  distil3d_select_gpu_resources "${DISTIL3D_SELECTED_GPU_COUNT}" || return $?
  if [[ -n "${resume_record}" ]]; then
    source_arguments=(--resume-record "${resume_record}" --selected-checkpoint "${selected_checkpoint}")
    control_command=(
      docker run --rm --network none
      --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs,readonly"
      --entrypoint "" "${DISTIL3D_SELECTED_IMAGE_ID}"
    )
  else
    source_arguments=(--job "${job_file}")
    control_command=(distil3d_run_with_config "${DISTIL3D_SELECTED_IMAGE_ID}")
  fi

  plan_output="$(
    "${control_command[@]}" \
    /opt/envs/core/bin/python \
    -m novel_view.cli.main \
    plan \
    "${source_arguments[@]}" \
    --config-root /project-config \
    --selected-preset "${DISTIL3D_SELECTED_PRESET}" \
    --image-variant "${DISTIL3D_SELECTED_IMAGE_VARIANT}" \
    --image-id "${DISTIL3D_SELECTED_IMAGE_ID}" \
    --image-source-revision "${DISTIL3D_SELECTED_IMAGE_SOURCE_REVISION}" \
    --image-source-dirty "${DISTIL3D_SELECTED_IMAGE_SOURCE_DIRTY}" \
    --image-lock-revision "${DISTIL3D_SELECTED_IMAGE_LOCK_REVISION}" \
    --host-data-root "${DISTIL3D_DATA_ROOT}" \
    --host-models-root "${DISTIL3D_MODELS_ROOT}" \
    --host-prepared-root "${DISTIL3D_PREPARED_ROOT}" \
    --host-runs-root "${DISTIL3D_RUNS_ROOT}" \
    --host-cache-root "${DISTIL3D_CACHE_ROOT}" \
    --host-jobs-root "${DISTIL3D_REPOSITORY_ROOT}/jobs" \
    --host-recipes-root "${DISTIL3D_REPOSITORY_ROOT}/recipes" \
    --host-selections-root "${DISTIL3D_REPOSITORY_ROOT}/selections" \
    --host-gpu-ids "${DISTIL3D_SELECTED_HOST_GPU_IDS}" \
    --container-gpu-indices "${DISTIL3D_SELECTED_CONTAINER_GPU_INDICES}" \
    --ipc="${DISTIL3D_SELECTED_IPC}" \
    --memlock="${DISTIL3D_SELECTED_MEMLOCK}" \
    --host-uid "$(id -u)" \
    --host-gid "$(id -g)"
  )" || return $?
  current_image_id="$(
    docker image inspect --format '{{.Id}}' "${DISTIL3D_SELECTED_IMAGE_TAG}"
  )" || return $?
  if [[ "${current_image_id}" != "${DISTIL3D_SELECTED_IMAGE_ID}" ]]; then
    printf 'selected image tag changed while resolving the job\n' >&2
    return 2
  fi
  printf '%s\n' "${plan_output}"
}
