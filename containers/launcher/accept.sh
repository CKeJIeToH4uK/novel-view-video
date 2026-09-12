#!/usr/bin/env bash

readonly DISTIL3D_ACCEPT_SUMMARY_HEADER='case_id	phase	required	status	recorded_state	worker_exit_code	container_state	container_exit_code	oom_killed	started	finished	attempt_ref	image_variant	image_id	evidence_bytes	evidence_over_budget	external_log_bytes	external_log_locator	note'
readonly DISTIL3D_ACCEPT_CASE_BUDGET_BYTES=26214400

_distil3d_accept_usage() {
  cat <<'EOF'
Usage:
  ./distil3d accept a100-4 --candidate <bundle/candidate.json> --profile <profile-directory> --output <evidence-directory>
EOF
}

_distil3d_accept_write_result() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$@" >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/summary.tsv"
}

_distil3d_accept_copy_artifact() {
  local source_file="$1"
  local destination_root="$2"
  local allowed_name="$3"
  local maximum_bytes="$4"
  local actual_bytes

  case "${allowed_name}" in
    attempt.tsv | record.json | metrics.json | metrics.csv | summary.csv | \
      preview.png | preview.mp4 | junit.xml | \
      resolved-job.json | plan.json | \
      container-inspect.json | gpu-map.tsv) ;;
    *)
      printf 'artifact is not allowlisted: %s\n' "${allowed_name}" >&2
      return 2
      ;;
  esac
  actual_bytes="$(wc -c < "${source_file}")" || return $?
  if (( actual_bytes > maximum_bytes )); then
    printf 'artifact exceeds its case budget: %s\n' "${allowed_name}" >&2
    return 2
  fi
  mkdir -p -- "${destination_root}"
  cp -- "${source_file}" "${destination_root}/${allowed_name}"
}

_distil3d_accept_parse_candidate_output() {
  local output="$1"
  local key
  local value
  local seen='|'

  DISTIL3D_ACCEPT_CANDIDATE_ID=''
  DISTIL3D_ACCEPT_SOURCE_REVISION=''
  DISTIL3D_ACCEPT_LOCK_REVISION=''
  DISTIL3D_ACCEPT_CORE_IMAGE_ID=''
  DISTIL3D_ACCEPT_MOGE_IMAGE_ID=''
  while IFS='=' read -r key value; do
    if [[ -z "${key}" || "${seen}" == *"|${key}|"* ]]; then
      printf 'candidate reader returned an empty or duplicate field\n' >&2
      return 2
    fi
    case "${key}" in
      candidate_id) DISTIL3D_ACCEPT_CANDIDATE_ID="${value}" ;;
      source_revision) DISTIL3D_ACCEPT_SOURCE_REVISION="${value}" ;;
      source_dirty) [[ "${value}" == false ]] || return 2 ;;
      platform) [[ "${value}" == linux/amd64 ]] || return 2 ;;
      lock_revision) DISTIL3D_ACCEPT_LOCK_REVISION="${value}" ;;
      core_tag) [[ "${value}" == "${DISTIL3D_CORE_IMAGE}" ]] || return 2 ;;
      core_image_id) DISTIL3D_ACCEPT_CORE_IMAGE_ID="${value}" ;;
      moge_tag) [[ "${value}" == "${DISTIL3D_MOGE_IMAGE}" ]] || return 2 ;;
      moge_image_id) DISTIL3D_ACCEPT_MOGE_IMAGE_ID="${value}" ;;
      transport) [[ "${value}" == docker-save ]] || return 2 ;;
      *)
        printf 'candidate reader returned an unknown field: %s\n' "${key}" >&2
        return 2
        ;;
    esac
    seen+="${key}|"
  done <<< "${output}"
  for key in candidate_id source_revision source_dirty platform lock_revision \
    core_tag core_image_id moge_tag moge_image_id transport
  do
    if [[ "${seen}" != *"|${key}|"* ]]; then
      printf 'candidate reader omitted a required field: %s\n' "${key}" >&2
      return 2
    fi
  done
}

_distil3d_accept_initialize_report() {
  local candidate_file="$1"
  local final_root

  final_root="${DISTIL3D_ACCEPT_OUTPUT_ROOT}/${DISTIL3D_ACCEPT_INVOCATION_ID}-${DISTIL3D_ACCEPT_CANDIDATE_ID}"
  mv -- "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}" "${final_root}"
  DISTIL3D_ACCEPT_CAMPAIGN_ROOT="${final_root}"
  mkdir -p -- "${final_root}/host" "${final_root}/images" \
    "${final_root}/plans" "${final_root}/cases" "${final_root}/records"
  cp -- "${candidate_file}" "${final_root}/candidate.json"
  printf '%b\n' "${DISTIL3D_ACCEPT_SUMMARY_HEADER}" > "${final_root}/summary.tsv"
  printf 'case_id\tphase\tcommand\n' > "${final_root}/commands.tsv"
}

_distil3d_accept_write_image_facts() {
  local variant="$1"
  local image_id="$2"
  local image_user

  image_user="$(docker image inspect --format '{{.Config.User}}' "${image_id}")" || return $?
  if [[ "${image_user}" != distil3d ]]; then
    printf 'candidate image does not use the distil3d runtime user: %s\n' \
      "${variant}" >&2
    return 2
  fi
  printf '{\n  "candidate_id": "%s",\n  "id": "%s",\n  "platform": "linux/amd64",\n  "source_revision": "%s",\n  "source_dirty": false,\n  "lock_revision": "%s",\n  "variant": "%s",\n  "gen3c_revision": "db2ffe12ced12ddafcec5e0422ee46ce8520746b",\n  "dinov2_revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",\n  "user": "distil3d"\n}\n' \
    "${DISTIL3D_ACCEPT_CANDIDATE_ID}" "${image_id}" \
    "${DISTIL3D_ACCEPT_SOURCE_REVISION}" "${DISTIL3D_ACCEPT_LOCK_REVISION}" \
    "${variant}" > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/images/${variant}-inspect.json"
}

_distil3d_accept_bootstrap() {
  local candidate_argument="$1"
  local candidate_directory
  local candidate_file
  local archive_file
  local candidate_output
  local actual_core_id
  local actual_moge_id
  local host_revision
  local docker_version
  local docker_architecture
  local docker_storage
  local command_status

  candidate_directory="$(CDPATH= cd -- "$(dirname -- "${candidate_argument}")" && pwd -P)" || return $?
  candidate_file="${candidate_directory}/$(basename -- "${candidate_argument}")"
  archive_file="${candidate_directory}/images.tar"
  if docker load --input "${archive_file}"; then
    :
  else
    command_status=$?
    printf 'candidate archive could not be loaded\n' >&2
    return "${command_status}"
  fi
  if candidate_output="$(
    docker run --rm --network none \
      --mount "type=bind,src=${candidate_file},dst=/candidate.json,readonly" \
      --entrypoint "" "${DISTIL3D_CORE_IMAGE}" \
      /opt/envs/core/bin/python -m novel_view.cli.candidate read \
      --candidate /candidate.json
  )"
  then
    :
  else
    command_status=$?
    printf 'candidate.json could not be parsed by the loaded core image\n' >&2
    return "${command_status}"
  fi
  _distil3d_accept_parse_candidate_output "${candidate_output}" || return $?
  _distil3d_accept_initialize_report "${candidate_file}" || return $?

  host_revision="$(git -C "${DISTIL3D_REPOSITORY_ROOT}" rev-parse HEAD)" || return $?
  if [[ "${host_revision}" != "${DISTIL3D_ACCEPT_SOURCE_REVISION}" || \
    "$(distil3d_source_dirty)" != false ]]
  then
    printf 'host checkout does not match the clean candidate source\n' >&2
    return 2
  fi

  actual_core_id="$(docker image inspect --format '{{.Id}}' "${DISTIL3D_CORE_IMAGE}")" || return $?
  actual_moge_id="$(docker image inspect --format '{{.Id}}' "${DISTIL3D_MOGE_IMAGE}")" || return $?
  if [[ "${actual_core_id}" != "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}" || \
    "${actual_moge_id}" != "${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" ]]
  then
    printf 'loaded image IDs do not match candidate.json\n' >&2
    return 2
  fi
  distil3d_check_candidate_image "${actual_core_id}" core \
    "${DISTIL3D_ACCEPT_SOURCE_REVISION}" "${DISTIL3D_ACCEPT_LOCK_REVISION}" || return $?
  distil3d_check_candidate_image "${actual_moge_id}" moge \
    "${DISTIL3D_ACCEPT_SOURCE_REVISION}" "${DISTIL3D_ACCEPT_LOCK_REVISION}" || return $?
  _distil3d_accept_write_image_facts core "${actual_core_id}" || return $?
  _distil3d_accept_write_image_facts moge "${actual_moge_id}" || return $?

  docker_version="$(docker version --format '{{.Server.Version}}')" || return $?
  docker_architecture="$(docker info --format '{{.Architecture}}')" || return $?
  docker_storage="$(docker info --format '{{.Driver}}')" || return $?
  printf 'source_revision\t%s\nsource_dirty\tfalse\n' "${host_revision}" \
    > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/source.tsv"
  printf 'docker.version=%s\ndocker.architecture=%s\ndocker.storage_driver=%s\n' \
    "${docker_version}" "${docker_architecture}" "${docker_storage}" \
    > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/docker-info.txt"
  printf 'locator\tbundle/images.tar\nbytes\t%s\nsha256\t%s\n' \
    "$(wc -c < "${archive_file}")" "$(distil3d_sha256_file "${archive_file}")" \
    > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/images/archive.tsv"
  printf 'bootstrap\tbootstrap\tdocker load sibling images.tar; strict candidate read in core; exact image inspect\n' \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
}

_distil3d_accept_case_key() {
  printf '%s\n' "${1//:/-}"
}

_distil3d_accept_note_failure() {
  if [[ "$1" == true ]]; then
    DISTIL3D_ACCEPT_REQUIRED_FAILED=1
  fi
}

_distil3d_accept_write_no_attempt() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local status="$4"
  local exit_code="$5"
  local note="$6"
  local moment

  moment="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  _distil3d_accept_write_result "${case_id}" "${phase}" "${required}" \
    "${status}" absent "${exit_code}" absent null false "${moment}" \
    "${moment}" none none none 0 false 0 none "${note}"
  if [[ "${status}" != succeeded ]]; then
    _distil3d_accept_note_failure "${required}"
  fi
}

_distil3d_accept_skip() {
  _distil3d_accept_write_no_attempt "$1" "$2" "$3" skipped 0 "$4"
}

_distil3d_accept_known_bytes() {
  local relative
  local total=0
  local file_bytes

  for relative in bootstrap.log candidate.json commands.tsv summary.tsv \
    host/source.tsv host/docker-info.txt images/archive.tsv \
    images/core-inspect.json images/moge-inspect.json
  do
    if [[ -f "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/${relative}" ]]; then
      file_bytes="$(wc -c < "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/${relative}")"
      total=$((total + file_bytes))
    fi
  done
  printf '%s\n' "${total}"
}

_distil3d_accept_start_te_recompute() {
  local case_id="$1"
  local attempt_id
  local evidence_root
  local -a command

  distil3d_load_profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" || return $?
  distil3d_select_gpu_resources 1 || return $?
  attempt_id="$(distil3d_new_id)"
  DISTIL3D_ACCEPT_ACTIVE_REF="native-te-recompute/${DISTIL3D_ACCEPT_INVOCATION_ID}/${attempt_id}"
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_VARIANT=core
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_ID="${DISTIL3D_ACCEPT_CORE_IMAGE_ID}"
  evidence_root="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/records/$(_distil3d_accept_case_key "${case_id}")"
  mkdir -p -- "${evidence_root}"
  command=(docker create --name "distil3d-${attempt_id}"
    --label io.distil3d.job=native-te-recompute
    --label "io.distil3d.run-id=${DISTIL3D_ACCEPT_INVOCATION_ID}"
    --label "io.distil3d.attempt-id=${attempt_id}"
    --label "io.distil3d.attempt-ref=${DISTIL3D_ACCEPT_ACTIVE_REF}"
    --user "$(id -u):$(id -g)" "${DISTIL3D_RESOURCE_ARGUMENTS[@]}"
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/diffusion/code/tests,dst=/tests,readonly"
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/diffusion/code/pyproject.toml,dst=/pyproject.toml,readonly"
    --mount "type=bind,src=${evidence_root},dst=/evidence"
    --entrypoint "" "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}"
    /opt/envs/gen3c/bin/python -m pytest -c /pyproject.toml -p no:cacheprovider -q
    /tests/integration/test_model_api.py::test_te_recompute_preserves_result_and_gradients
    --junitxml=/evidence/junit.xml)
  {
    printf '%s\tnative-api\t' "${case_id}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
  DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="$("${command[@]}")" || return $?
  _distil3d_accept_start_container
}

_distil3d_accept_te_recompute() {
  local case_id=ac:test:tc:model:test-gen3c-activation-checkpointing:gen3c-activation-checkpointing-tests
  local started
  local command_status

  _distil3d_accept_begin_active "${case_id}" native-te-recompute
  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if _distil3d_accept_start_te_recompute "${case_id}" \
    >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
  then
    _distil3d_accept_finish_active "${case_id}" native-api true "${started}"
    return
  else
    command_status=$?
  fi
  _distil3d_accept_start_failed "${case_id}" native-api true "${started}" "${command_status}"
}

accept_a100_4() {
  local candidate_argument=''
  local profile_argument=''
  local output_argument=''
  local started
  local finished
  local bootstrap_status
  local evidence_bytes
  local campaign_status

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --candidate | --profile | --output)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          printf 'missing value for %s\n' "$1" >&2
          return 2
        fi
        case "$1" in
          --candidate) candidate_argument="$2" ;;
          --profile) profile_argument="$2" ;;
          --output) output_argument="$2" ;;
        esac
        shift 2
        ;;
      -h | --help)
        _distil3d_accept_usage
        return 0
        ;;
      *)
        printf 'unknown accept a100-4 argument: %s\n' "$1" >&2
        return 2
        ;;
    esac
  done
  if [[ -z "${candidate_argument}" || -z "${profile_argument}" || \
    -z "${output_argument}" ]]
  then
    printf 'accept a100-4 requires --candidate, --profile and --output\n' >&2
    return 2
  fi

  mkdir -p -- "${output_argument}"
  DISTIL3D_ACCEPT_OUTPUT_ROOT="$(CDPATH= cd -- "${output_argument}" && pwd -P)"
  DISTIL3D_ACCEPT_PROFILE_ARGUMENT="${profile_argument}"
  DISTIL3D_ACCEPT_INVOCATION_ID="$(distil3d_new_id)"
  DISTIL3D_ACCEPT_CAMPAIGN_ROOT="${DISTIL3D_ACCEPT_OUTPUT_ROOT}/${DISTIL3D_ACCEPT_INVOCATION_ID}-bootstrap"
  mkdir -m 700 -- "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}"
  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'campaign_bootstrap=%s\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}"

  if _distil3d_accept_bootstrap "${candidate_argument}" \
    > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/bootstrap.log" 2>&1
  then
    bootstrap_status=0
  else
    bootstrap_status=$?
  fi
  finished="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if [[ ${bootstrap_status} -ne 0 ]]; then
    if [[ -f "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/summary.tsv" ]]; then
      evidence_bytes="$(_distil3d_accept_known_bytes)"
      _distil3d_accept_write_result bootstrap bootstrap true failed absent \
        "${bootstrap_status}" absent null false "${started}" "${finished}" \
        none none none "${evidence_bytes}" false 0 none 'bootstrap failed'
    fi
    cat "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/bootstrap.log" >&2
    printf 'campaign=%s\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}" >&2
    return "${bootstrap_status}"
  fi

  evidence_bytes="$(_distil3d_accept_known_bytes)"
  _distil3d_accept_write_result bootstrap bootstrap true succeeded succeeded 0 \
    absent 0 false "${started}" "${finished}" none core+moge \
    "${DISTIL3D_ACCEPT_CORE_IMAGE_ID},${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" \
    "${evidence_bytes}" false 0 none 'candidate verified'

  DISTIL3D_ACCEPT_REQUIRED_FAILED=0
  DISTIL3D_ACCEPT_INTERRUPTED=0
  DISTIL3D_ACCEPT_ABORT=0
  DISTIL3D_ACCEPT_FOLLOWER_PID=''
  DISTIL3D_ACCEPT_ACTIVE_REF=''
  DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID=''
  DISTIL3D_ACCEPT_ACTIVE_LOG=/dev/null
  DISTIL3D_ACCEPT_V1_SELECTED_RECORD=''
  DISTIL3D_ACCEPT_V1_SELECTED_STEP=''
  DISTIL3D_ACCEPT_V2_SELECTED_RECORD=''
  DISTIL3D_ACCEPT_V2_SELECTED_STEP=''
  trap '_distil3d_accept_on_signal' INT TERM

  if _distil3d_accept_prepare_jobs; then
    :
  else
    campaign_status=$?
    _distil3d_accept_write_no_attempt foundation foundation true failed \
      "${campaign_status}" 'campaign-local forms could not be prepared'
    _distil3d_accept_close_unstarted 'skipped:first-failed(foundation)'
    _distil3d_accept_closure || true
    trap - INT TERM
    cat "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/bootstrap.log"
    printf 'campaign=%s\nacceptance=failed\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}" >&2
    return "${campaign_status}"
  fi
  if _distil3d_accept_foundation; then
    :
  else
    campaign_status=$?
    _distil3d_accept_close_unstarted 'skipped:first-failed(foundation)'
    _distil3d_accept_closure || true
    trap - INT TERM
    cat "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/bootstrap.log"
    printf 'campaign=%s\nacceptance=failed\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}" >&2
    return "${campaign_status}"
  fi
  if _distil3d_accept_campaign; then
    campaign_status=0
  else
    campaign_status=$?
  fi
  trap - INT TERM
  cat "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/bootstrap.log"
  if [[ ${campaign_status} -eq 0 ]]; then
    printf 'campaign=%s\nacceptance=succeeded\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}"
  else
    printf 'campaign=%s\nacceptance=failed\n' "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}" >&2
  fi
  return "${campaign_status}"
}
