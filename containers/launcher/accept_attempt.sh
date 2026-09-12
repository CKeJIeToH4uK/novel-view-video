#!/usr/bin/env bash

_distil3d_accept_record_reader() {
  local kind="$1"
  local version="$2"
  local path="$3"
  local expected_attempt="$4"

  docker run --rm --network none \
    --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs,readonly" \
    --entrypoint "" "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}" \
    /opt/envs/core/bin/python -m novel_view.cli.acceptance "${kind}" \
    --version "${version}" --path "${path}" \
    --expected-attempt "${expected_attempt}"
}

_distil3d_accept_attempt_locator() {
  distil3d_parse_attempt_reference "$1" || return $?
  printf '%s/%s/attempts/%s\n' "${DISTIL3D_REFERENCE_JOB_NAME}" \
    "${DISTIL3D_REFERENCE_RUN_ID}" "${DISTIL3D_REFERENCE_ATTEMPT_ID}"
}

_distil3d_accept_read_checkpoint() {
  local version="$1"
  local attempt_ref="$2"
  local output
  local key
  local value
  local seen='|'

  output="$(_distil3d_accept_record_reader checkpoint "${version}" \
    "/runs/$(_distil3d_accept_attempt_locator "${attempt_ref}")/checkpoints/step-000000077.json" \
    "${attempt_ref}")" \
    || return $?
  DISTIL3D_ACCEPT_CHECKPOINT=''
  while IFS='=' read -r key value; do
    case "${key}" in
      source_attempt) [[ "${value}" == "${attempt_ref}" ]] || return 2 ;;
      completed_step) [[ "${value}" == 77 ]] || return 2 ;;
      checkpoint) DISTIL3D_ACCEPT_CHECKPOINT="${value}" ;;
      *) return 2 ;;
    esac
    [[ "${seen}" != *"|${key}|"* ]] || return 2
    seen+="${key}|"
  done <<< "${output}"
  [[ "${seen}" == *'|source_attempt|'* && "${seen}" == *'|completed_step|'* && \
    "${seen}" == *'|checkpoint|'* && -n "${DISTIL3D_ACCEPT_CHECKPOINT}" ]]
}

_distil3d_accept_read_selection() {
  local version="$1"
  local selection_attempt="$2"
  local training_attempt="$3"
  local output
  local key
  local value
  local seen='|'

  output="$(_distil3d_accept_record_reader selection "${version}" \
    "/runs/$(_distil3d_accept_attempt_locator "${selection_attempt}")/selection.json" \
    "${training_attempt}")" || return $?
  DISTIL3D_ACCEPT_SELECTED_RECORD=''
  DISTIL3D_ACCEPT_SELECTED_CHECKPOINT=''
  DISTIL3D_ACCEPT_SELECTED_STEP=''
  while IFS='=' read -r key value; do
    case "${key}" in
      training_attempt) [[ "${value}" == "${training_attempt}" ]] || return 2 ;;
      completed_step) DISTIL3D_ACCEPT_SELECTED_STEP="${value}" ;;
      checkpoint_record) DISTIL3D_ACCEPT_SELECTED_RECORD="${value}" ;;
      checkpoint) DISTIL3D_ACCEPT_SELECTED_CHECKPOINT="${value}" ;;
      *) return 2 ;;
    esac
    [[ "${seen}" != *"|${key}|"* ]] || return 2
    seen+="${key}|"
  done <<< "${output}"
  [[ "${seen}" == *'|training_attempt|'* && "${seen}" == *'|completed_step|'* && \
    "${seen}" == *'|checkpoint_record|'* && "${seen}" == *'|checkpoint|'* ]]
}

_distil3d_accept_read_waymo_result() {
  local case_id="$1"
  local attempt_root="/runs/$(_distil3d_accept_attempt_locator "${DISTIL3D_ACCEPT_ACTIVE_REF}")"
  local selections=/project-config/selections/local/stage11/legacy
  local kind
  local clip
  local output
  local key
  local value
  local seen='|'
  local -a arguments=()

  case "${case_id}" in
    ac:scenario:sc:waymo-depth-candidates-v2)
      arguments=(waymo-candidate --path "${attempt_root}/result.json"
        --clip-selection "${selections}/waymo-depth-clip.yaml") ;;
    ac:scenario:sc:waymo-depth-selection-gate-v2)
      arguments=(waymo-depth-selection --path "${attempt_root}/depth-selection.json"
        --reports-selection "${selections}/waymo-depth-choice-8.yaml") ;;
    *)
      case "${case_id}" in
        ac:scenario:sc:waymo-ddw-engineering-canary-v1)
          kind=canary-v1; clip=waymo-ddw-exposed-debug ;;
        ac:scenario:sc:waymo-ddw-v2-canary-v1)
          kind=canary-v2; clip=waymo-ddw-canary ;;
        ac:scenario:sc:waymo-ddw-v3-probe-v1)
          kind=probe; clip=waymo-ddw-canary ;;
        ac:scenario:sc:waymo-ddw-survey-v1)
          kind=survey; clip=waymo-ddw-survey ;;
      esac
      arguments=(waymo-ddw --kind "${kind}" --path "${attempt_root}/result.json"
        --clip-selection "${selections}/${clip}.yaml"
        --expected-depth-selection "${DISTIL3D_ACCEPT_DEPTH_SELECTION}") ;;
  esac
  output="$(docker run --rm --network none \
    --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs,readonly" \
    --mount "type=bind,src=${DISTIL3D_REPOSITORY_ROOT}/selections,dst=/project-config/selections,readonly" \
    --entrypoint "" "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}" \
    /opt/envs/core/bin/python -m novel_view.cli.acceptance "${arguments[@]}")" || return $?
  DISTIL3D_ACCEPT_WAYMO_FORMAT=''
  DISTIL3D_ACCEPT_WAYMO_OUTCOME=''
  DISTIL3D_ACCEPT_WAYMO_EXPECTED_EXIT=''
  DISTIL3D_ACCEPT_WAYMO_BACKEND=''
  DISTIL3D_ACCEPT_WAYMO_DDW_ALLOWED=''
  while IFS='=' read -r key value; do
    case "${key}" in
      format) DISTIL3D_ACCEPT_WAYMO_FORMAT="${value}" ;;
      outcome) DISTIL3D_ACCEPT_WAYMO_OUTCOME="${value}" ;;
      expected_exit) DISTIL3D_ACCEPT_WAYMO_EXPECTED_EXIT="${value}" ;;
      selected_backend) DISTIL3D_ACCEPT_WAYMO_BACKEND="${value}" ;;
      ddw_allowed) DISTIL3D_ACCEPT_WAYMO_DDW_ALLOWED="${value}" ;;
      *) return 2 ;;
    esac
    [[ "${seen}" != *"|${key}|"* ]] || return 2
    seen+="${key}|"
  done <<< "${output}"
  [[ "${seen}" == *'|format|'* && "${seen}" == *'|outcome|'* && \
    "${seen}" == *'|expected_exit|'* && "${seen}" == *'|selected_backend|'* && \
    "${seen}" == *'|ddw_allowed|'* ]]
}

_distil3d_accept_write_case_files() {
  local case_root="$1"
  local log_file="${case_root}/output.log"
  local excerpt="${case_root}/output.excerpt"
  local log_digest

  DISTIL3D_ACCEPT_LOG_BYTES=0
  DISTIL3D_ACCEPT_LOG_LOCATOR=none
  DISTIL3D_ACCEPT_EVIDENCE_OVER=false
  if [[ -f "${log_file}" ]]; then
    DISTIL3D_ACCEPT_LOG_BYTES="$(wc -c < "${log_file}")"
    log_digest="$(distil3d_sha256_file "${log_file}")"
    DISTIL3D_ACCEPT_LOG_LOCATOR="cases/$(basename -- "${case_root}")/output.log"
    if (( DISTIL3D_ACCEPT_LOG_BYTES + DISTIL3D_ACCEPT_RECORD_BYTES > \
      DISTIL3D_ACCEPT_CASE_BUDGET_BYTES )); then
      head -c 65536 "${log_file}" > "${excerpt}"
      printf '\n--- omitted ---\n' >> "${excerpt}"
      tail -c 65536 "${log_file}" >> "${excerpt}"
      mv -- "${excerpt}" "${log_file}"
      DISTIL3D_ACCEPT_LOG_LOCATOR="docker:${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}"
      DISTIL3D_ACCEPT_EVIDENCE_OVER=true
    fi
    printf 'bytes\t%s\nsha256\t%s\nlocator\t%s\n' \
      "${DISTIL3D_ACCEPT_LOG_BYTES}" "${log_digest}" \
      "${DISTIL3D_ACCEPT_LOG_LOCATOR}" > "${case_root}/output.meta.tsv"
  fi
  DISTIL3D_ACCEPT_CASE_BYTES="${DISTIL3D_ACCEPT_RECORD_BYTES}"
  for file in output.log output.meta.tsv attempt.tsv container-inspect.json gpu-map.tsv plan.log; do
    if [[ -f "${case_root}/${file}" ]]; then
      DISTIL3D_ACCEPT_CASE_BYTES=$((
        DISTIL3D_ACCEPT_CASE_BYTES + $(wc -c < "${case_root}/${file}")
      ))
    fi
  done
}

_distil3d_accept_write_container_facts() {
  local case_root="$1"

  printf 'attempt_ref\trecorded_state\trecord_present\n%s\t%s\t%s\n' \
    "${DISTIL3D_ACCEPT_ACTIVE_REF}" "${DISTIL3D_RECORDED_STATE}" \
    "${DISTIL3D_RECORD_PRESENT}" > "${case_root}/attempt.tsv"
  printf '{\n  "id": "%s",\n  "state": "%s",\n  "exit_code": %s,\n  "oom_killed": %s\n}\n' \
    "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" "${DISTIL3D_CONTAINER_STATE}" \
    "${DISTIL3D_CONTAINER_EXIT_CODE}" "${DISTIL3D_CONTAINER_OOM_KILLED}" \
    > "${case_root}/container-inspect.json"
  printf 'host_gpu_ids\tcontainer_gpu_indices\n%s\t%s\n' \
    "${DISTIL3D_SELECTED_HOST_GPU_IDS}" "${DISTIL3D_SELECTED_CONTAINER_GPU_INDICES}" \
    > "${case_root}/gpu-map.tsv"
}

_distil3d_accept_copy_records() {
  local case_id="$1"
  local attempt_root
  local destination
  local name

  DISTIL3D_ACCEPT_RECORD_BYTES=0
  attempt_root="${DISTIL3D_RUNS_ROOT}/$(_distil3d_accept_attempt_locator \
    "${DISTIL3D_ACCEPT_ACTIVE_REF}")"
  destination="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/records/$(_distil3d_accept_case_key \
    "${case_id}")"
  case "${case_id}" in
    ac:scenario:sc:euvs-base-tuned-comparison-v1)
      _distil3d_accept_copy_artifact "${attempt_root}/summary.csv" \
        "${destination}" summary.csv 4194304 || return $?
      ;;
    ac:scenario:sc:gen3c-checkpoint-selection-v1 | \
      ac:scenario:sc:gen3c-checkpoint-selection-v2)
      _distil3d_accept_copy_artifact "${attempt_root}/selection.json" \
        "${destination}" record.json 4194304 \
        || return $?
      ;;
    ac:scenario:sc:gen3c-ddw-evaluation-v2)
      _distil3d_accept_copy_artifact \
        "${attempt_root}/ddw-heldout-point-evaluation.json" \
        "${destination}" record.json 4194304 \
        || return $?
      ;;
    ac:scenario:sc:waymo-depth-selection-gate-v2)
      _distil3d_accept_copy_artifact "${attempt_root}/depth-selection.json" \
        "${destination}" record.json 4194304 || return $?
      ;;
    ac:scenario:sc:waymo-depth-candidates-v2 | \
      ac:scenario:sc:waymo-ddw-engineering-canary-v1 | \
      ac:scenario:sc:waymo-ddw-v2-canary-v1 | \
      ac:scenario:sc:waymo-ddw-v3-probe-v1 | \
      ac:scenario:sc:waymo-ddw-survey-v1)
      _distil3d_accept_copy_artifact "${attempt_root}/result.json" \
        "${destination}" record.json 4194304 || return $?
      ;;
  esac
  if [[ -d "${destination}" ]]; then
    for name in "${destination}"/*; do
      DISTIL3D_ACCEPT_RECORD_BYTES=$((
        DISTIL3D_ACCEPT_RECORD_BYTES + $(wc -c < "${name}")
      ))
    done
  fi
}

_distil3d_accept_stop_active() {
  local resolved=''

  if [[ -z "${DISTIL3D_ACCEPT_ACTIVE_REF}" && \
    -n "${DISTIL3D_ATTEMPT_REF:-}" ]]
  then
    DISTIL3D_ACCEPT_ACTIVE_REF="${DISTIL3D_ATTEMPT_REF}"
    DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="${DISTIL3D_ATTEMPT_CONTAINER_ID:-}"
  fi
  if [[ -z "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" && \
    -n "${DISTIL3D_ACCEPT_ACTIVE_REF}" ]]
  then
    if distil3d_find_exact_container "${DISTIL3D_ACCEPT_ACTIVE_REF}"; then
      resolved="${DISTIL3D_RESOLVED_CONTAINER_ID}"
      DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="${resolved}"
    else
      return $?
    fi
  fi
  if [[ -z "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" ]]; then
    DISTIL3D_CONTAINER_STATE=absent
    return 0
  fi
  if distil3d_inspect_container "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}"; then
    :
  else
    return $?
  fi
  case "${DISTIL3D_CONTAINER_STATE}" in
    created)
      if docker rm "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" >/dev/null; then
        DISTIL3D_CONTAINER_STATE=absent
        DISTIL3D_CONTAINER_EXIT_CODE=null
        DISTIL3D_CONTAINER_OOM_KILLED=false
        return 0
      else
        return $?
      fi
      ;;
    running | paused | restarting)
      if distil3d_stop_main --profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" \
        "${DISTIL3D_ACCEPT_ACTIVE_REF}" --grace-seconds 10
      then
        distil3d_inspect_container "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" || return $?
        [[ "${DISTIL3D_CONTAINER_STATE}" == exited || \
          "${DISTIL3D_CONTAINER_STATE}" == dead ]]
        return
      else
        return $?
      fi
      ;;
    exited | dead | absent)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_distil3d_accept_on_signal() {
  trap '' INT TERM
  DISTIL3D_ACCEPT_INTERRUPTED=1
  DISTIL3D_ACCEPT_ABORT=1
  if [[ -n "${DISTIL3D_ACCEPT_FOLLOWER_PID}" ]]; then
    kill -TERM "${DISTIL3D_ACCEPT_FOLLOWER_PID}" 2>/dev/null || true
  fi
  if _distil3d_accept_stop_active >> "${DISTIL3D_ACCEPT_ACTIVE_LOG:-/dev/null}" 2>&1; then
    DISTIL3D_ACCEPT_CLEANUP_CONFIRMED=true
  else
    DISTIL3D_ACCEPT_CLEANUP_CONFIRMED=false
  fi
}

_distil3d_accept_begin_active() {
  local case_id="$1"
  local name="$2"

  DISTIL3D_ACCEPT_ACTIVE_CASE="${case_id}"
  DISTIL3D_ACCEPT_ACTIVE_NAME="${name}"
  DISTIL3D_ACCEPT_ACTIVE_REF=''
  DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID=''
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_VARIANT=none
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_ID=none
  DISTIL3D_ACCEPT_ACTIVE_RECORDED_STATE=absent
  DISTIL3D_ACCEPT_ACTIVE_LOG="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/cases/$(_distil3d_accept_case_key "${case_id}")/output.log"
  DISTIL3D_ACCEPT_FOLLOWER_PID=''
  DISTIL3D_ACCEPT_CLEANUP_CONFIRMED=true
  DISTIL3D_ACCEPT_RECORD_BYTES=0
  DISTIL3D_ATTEMPT_REF=''
  DISTIL3D_ATTEMPT_CONTAINER_ID=''
  DISTIL3D_CONTAINER_STATE=absent
  DISTIL3D_CONTAINER_EXIT_CODE=null
  DISTIL3D_CONTAINER_OOM_KILLED=false
  mkdir -p -- "$(dirname -- "${DISTIL3D_ACCEPT_ACTIVE_LOG}")"
}

_distil3d_accept_start_run() {
  local name="$1"
  local run_id="$2"
  local job_file

  distil3d_load_profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" || return $?
  job_file="$(distil3d_job_container_file "${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}/${name}")"
  distil3d_select_job_image "${job_file}" || return $?
  _distil3d_accept_require_selected_candidate || return $?
  distil3d_final_job_plan "${job_file}" >/dev/null || return $?
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_VARIANT="${DISTIL3D_SELECTED_IMAGE_VARIANT}"
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_ID="${DISTIL3D_SELECTED_IMAGE_ID}"
  distil3d_create_attempt run "${DISTIL3D_SELECTED_JOB_NAME}" "${run_id}" \
    "${job_file}" "" "" || return $?
  DISTIL3D_ACCEPT_ACTIVE_REF="${DISTIL3D_ATTEMPT_REF}"
  DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="${DISTIL3D_ATTEMPT_CONTAINER_ID}"
  _distil3d_accept_start_container
}

_distil3d_accept_start_resume() {
  local source_ref="$1"
  local checkpoint="$2"
  local job_name
  local run_id
  local source_attempt_id

  distil3d_parse_attempt_reference "${source_ref}" || return $?
  job_name="${DISTIL3D_REFERENCE_JOB_NAME}"
  run_id="${DISTIL3D_REFERENCE_RUN_ID}"
  source_attempt_id="${DISTIL3D_REFERENCE_ATTEMPT_ID}"
  distil3d_load_profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" || return $?
  distil3d_read_resume_record "${job_name}" "${run_id}" "${source_attempt_id}" \
    || return $?
  distil3d_inspect_selected_image || return $?
  _distil3d_accept_require_selected_candidate || return $?
  distil3d_final_job_plan "" "${DISTIL3D_SELECTED_RESUME_RECORD}" \
    "${checkpoint}" > "${DISTIL3D_ACCEPT_ACTIVE_RESUME_PLAN}" || return $?
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_VARIANT="${DISTIL3D_SELECTED_IMAGE_VARIANT}"
  DISTIL3D_ACCEPT_ACTIVE_IMAGE_ID="${DISTIL3D_SELECTED_IMAGE_ID}"
  distil3d_create_attempt resume "${job_name}" "${run_id}" "" \
    "${DISTIL3D_SELECTED_RESUME_RECORD}" "${checkpoint}" || return $?
  DISTIL3D_ACCEPT_ACTIVE_REF="${DISTIL3D_ATTEMPT_REF}"
  DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="${DISTIL3D_ATTEMPT_CONTAINER_ID}"
  _distil3d_accept_start_container
}

_distil3d_accept_start_container() {
  if [[ ${DISTIL3D_ACCEPT_INTERRUPTED} -eq 1 ]]; then return 130; fi
  docker start "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" >/dev/null || return $?
  if [[ ${DISTIL3D_ACCEPT_INTERRUPTED} -eq 1 ]]; then return 130; fi
}

_distil3d_accept_record_active() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local status="$4"
  local started="$5"
  local note="$6"
  local case_root
  local finished

  case_root="$(dirname -- "${DISTIL3D_ACCEPT_ACTIVE_LOG}")"
  finished="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  _distil3d_accept_write_case_files "${case_root}"
  _distil3d_accept_write_result "${case_id}" "${phase}" "${required}" \
    "${status}" "${DISTIL3D_ACCEPT_ACTIVE_RECORDED_STATE}" \
    "${DISTIL3D_CONTAINER_EXIT_CODE:-null}" \
    "${DISTIL3D_CONTAINER_STATE:-absent}" \
    "${DISTIL3D_CONTAINER_EXIT_CODE:-null}" \
    "${DISTIL3D_CONTAINER_OOM_KILLED:-false}" "${started}" "${finished}" \
    "${DISTIL3D_ACCEPT_ACTIVE_REF:-none}" \
    "${DISTIL3D_ACCEPT_ACTIVE_IMAGE_VARIANT}" \
    "${DISTIL3D_ACCEPT_ACTIVE_IMAGE_ID}" "${DISTIL3D_ACCEPT_CASE_BYTES}" \
    "${DISTIL3D_ACCEPT_EVIDENCE_OVER}" "${DISTIL3D_ACCEPT_LOG_BYTES}" \
    "${DISTIL3D_ACCEPT_LOG_LOCATOR}" "${note}"
  if [[ "${status}" != succeeded ]]; then
    _distil3d_accept_note_failure "${required}"
  fi
}

_distil3d_accept_finish_active() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local started="$4"
  local selection_version="${5:-}"
  local expected_training_attempt="${6:-}"
  local follower_status=0
  local case_root
  local junit_file

  case_root="$(dirname -- "${DISTIL3D_ACCEPT_ACTIVE_LOG}")"
  docker logs --follow "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}" \
    >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1 &
  DISTIL3D_ACCEPT_FOLLOWER_PID=$!
  if wait "${DISTIL3D_ACCEPT_FOLLOWER_PID}"; then
    follower_status=0
  else
    follower_status=$?
  fi
  DISTIL3D_ACCEPT_FOLLOWER_PID=''

  if distil3d_inspect_container "${DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID}"; then
    :
  else
    printf 'container inspect failed after log follower\n' \
      >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}"
    DISTIL3D_CONTAINER_STATE=unknown
    DISTIL3D_CONTAINER_EXIT_CODE=null
    DISTIL3D_CONTAINER_OOM_KILLED=false
    if _distil3d_accept_stop_active >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1; then
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        failed "${started}" 'inspect failed; exact cleanup confirmed'
    else
      DISTIL3D_ACCEPT_ABORT=1
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        interrupted "${started}" 'cleanup-unknown; use exact status/stop recovery'
    fi
    return 1
  fi

  if [[ ${DISTIL3D_ACCEPT_INTERRUPTED} -eq 1 ]]; then
    _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
      interrupted "${started}" 'campaign interrupted'
    return 130
  fi
  case "${DISTIL3D_CONTAINER_STATE}" in
    created | running | paused | restarting)
      printf 'log follower ended with active container; status=cleanup-unknown\n' \
        >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}"
      if _distil3d_accept_stop_active >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1; then
        _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
          failed "${started}" 'cleanup-unknown; exact cleanup confirmed'
      else
        DISTIL3D_ACCEPT_ABORT=1
        _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
          interrupted "${started}" 'cleanup-unknown; use exact status/stop recovery'
      fi
      return 1
      ;;
    exited | dead) ;;
    *)
      DISTIL3D_ACCEPT_ABORT=1
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        interrupted "${started}" 'unknown terminal container state'
      return 1
      ;;
  esac

  distil3d_parse_attempt_reference "${DISTIL3D_ACCEPT_ACTIVE_REF}" || return $?
  if distil3d_read_attempt_snapshot \
    "${DISTIL3D_REFERENCE_JOB_NAME}" "${DISTIL3D_REFERENCE_RUN_ID}" \
    "${DISTIL3D_REFERENCE_ATTEMPT_ID}" 2>> "${DISTIL3D_ACCEPT_ACTIVE_LOG}"
  then
    :
  else
    _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
      failed "${started}" 'attempt record could not be read'
    return 1
  fi
  if [[ "${DISTIL3D_RESOLVED_ATTEMPT_REF}" != "${DISTIL3D_ACCEPT_ACTIVE_REF}" ]]; then
    _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
      failed "${started}" 'attempt reader returned a mixed reference'
    return 1
  fi
  DISTIL3D_ACCEPT_ACTIVE_RECORDED_STATE="${DISTIL3D_RECORDED_STATE}"
  _distil3d_accept_write_container_facts "${case_root}"
  case "${case_id}" in
    ac:test:tc:model:test-gen3c-activation-checkpointing:gen3c-activation-checkpointing-tests)
      junit_file="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/records/$(basename -- "${case_root}")/junit.xml"
      if [[ -f "${junit_file}" ]]; then
        DISTIL3D_ACCEPT_RECORD_BYTES="$(wc -c < "${junit_file}")"
      fi
      if [[ "${DISTIL3D_CONTAINER_EXIT_CODE}" == 0 && \
        "${DISTIL3D_CONTAINER_OOM_KILLED}" == false ]]
      then
        _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
          succeeded "${started}" 'native TE recompute passed; no training record'
        return 0
      fi
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        failed "${started}" 'native TE recompute exit or OOM failed'
      return 1
      ;;
    ac:scenario:sc:waymo-depth-candidates-v2 | \
      ac:scenario:sc:waymo-depth-selection-gate-v2 | \
      ac:scenario:sc:waymo-ddw-engineering-canary-v1 | \
      ac:scenario:sc:waymo-ddw-v2-canary-v1 | \
      ac:scenario:sc:waymo-ddw-v3-probe-v1 | \
      ac:scenario:sc:waymo-ddw-survey-v1)
      if [[ "${DISTIL3D_RECORD_PRESENT}" == true && \
        "${DISTIL3D_CONTAINER_OOM_KILLED}" == false ]] && \
        _distil3d_accept_read_waymo_result "${case_id}" \
          >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1 && \
        [[ "${DISTIL3D_CONTAINER_EXIT_CODE}" == "${DISTIL3D_ACCEPT_WAYMO_EXPECTED_EXIT}" && \
          ( ( "${DISTIL3D_CONTAINER_EXIT_CODE}" == 0 && "${DISTIL3D_RECORDED_STATE}" == succeeded ) || \
            ( "${DISTIL3D_CONTAINER_EXIT_CODE}" == 2 && "${DISTIL3D_RECORDED_STATE}" == failed ) ) ]] && \
        _distil3d_accept_copy_records "${case_id}" >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
      then
        _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
          succeeded "${started}" \
          "${DISTIL3D_ACCEPT_WAYMO_FORMAT}; outcome=${DISTIL3D_ACCEPT_WAYMO_OUTCOME}; selected_backend=${DISTIL3D_ACCEPT_WAYMO_BACKEND}; ddw_allowed=${DISTIL3D_ACCEPT_WAYMO_DDW_ALLOWED}"
        return 0
      fi
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        failed "${started}" 'historical outcome, exit, OOM or bounded evidence did not match'
      return 1
      ;;
  esac
  if [[ "${DISTIL3D_CONTAINER_EXIT_CODE}" == 0 && \
    "${DISTIL3D_RECORDED_STATE}" == succeeded && \
    "${DISTIL3D_RECORD_PRESENT}" == true ]]
  then
    if [[ -n "${selection_version}" ]] && ! _distil3d_accept_read_selection \
      "${selection_version}" "${DISTIL3D_ACCEPT_ACTIVE_REF}" \
      "${expected_training_attempt}" >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
    then
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        failed "${started}" 'selection record could not be read'
      return 1
    fi
    if ! _distil3d_accept_copy_records "${case_id}" \
      >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
    then
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        failed "${started}" 'bounded acceptance record could not be copied'
      return 1
    fi
    _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
      succeeded "${started}" "terminal inspect succeeded; follower=${follower_status}"
    return 0
  fi
  _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
    failed "${started}" "terminal attempt failed; follower=${follower_status}"
  return 1
}

_distil3d_accept_start_failed() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local started="$4"
  local command_status="$5"

  if [[ -z "${DISTIL3D_ACCEPT_ACTIVE_REF}" && -n "${DISTIL3D_ATTEMPT_REF:-}" ]]; then
    DISTIL3D_ACCEPT_ACTIVE_REF="${DISTIL3D_ATTEMPT_REF}"
    DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID="${DISTIL3D_ATTEMPT_CONTAINER_ID:-}"
  fi
  if [[ ${DISTIL3D_ACCEPT_INTERRUPTED} -eq 1 ]]; then
    DISTIL3D_ACCEPT_ACTIVE_CONTAINER_ID=''
  fi
  if [[ -n "${DISTIL3D_ACCEPT_ACTIVE_REF}" ]]; then
    if _distil3d_accept_stop_active >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1; then
      :
    else
      DISTIL3D_ACCEPT_ABORT=1
      _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
        interrupted "${started}" 'start failed and exact cleanup is unknown'
      return 1
    fi
  fi
  if [[ ${DISTIL3D_ACCEPT_INTERRUPTED} -eq 1 ]]; then
    _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
      interrupted "${started}" 'campaign interrupted while starting attempt; cleanup confirmed'
    return 130
  fi
  _distil3d_accept_record_active "${case_id}" "${phase}" "${required}" \
    failed "${started}" "attempt start failed with ${command_status}"
  return "${command_status}"
}

_distil3d_accept_run_case() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local name="$4"
  local selection_version="${5:-}"
  local expected_training_attempt="${6:-}"
  local started
  local command_status
  local run_id="${DISTIL3D_ACCEPT_INVOCATION_ID}-${name}"

  _distil3d_accept_begin_active "${case_id}" "${name}"
  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s\trun\t./distil3d run --profile <profile> %s/%s --run-id %s --detach\n' \
    "${case_id}" "${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}" "${name}" "${run_id}" \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
  if _distil3d_accept_start_run "${name}" "${run_id}" \
    >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
  then
    _distil3d_accept_finish_active "${case_id}" "${phase}" "${required}" \
      "${started}" "${selection_version}" "${expected_training_attempt}"
    return
  else
    command_status=$?
  fi
  _distil3d_accept_start_failed "${case_id}" "${phase}" "${required}" \
    "${started}" "${command_status}"
}

_distil3d_accept_resume_case() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local name="$4"
  local source_ref="$5"
  local checkpoint="$6"
  local started
  local command_status
  local key

  _distil3d_accept_begin_active "${case_id}" "${name}"
  key="$(_distil3d_accept_case_key "${case_id}")"
  DISTIL3D_ACCEPT_ACTIVE_RESUME_PLAN="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/plans/${key}.json"
  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '%s\tresume\t./distil3d resume --profile <profile> %s --checkpoint %s --detach\n' \
    "${case_id}" "${source_ref}" "${checkpoint}" \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
  if _distil3d_accept_start_resume "${source_ref}" "${checkpoint}" \
    >> "${DISTIL3D_ACCEPT_ACTIVE_LOG}" 2>&1
  then
    _distil3d_accept_finish_active "${case_id}" "${phase}" "${required}" \
      "${started}"
    return
  else
    command_status=$?
  fi
  _distil3d_accept_start_failed "${case_id}" "${phase}" "${required}" \
    "${started}" "${command_status}"
}
