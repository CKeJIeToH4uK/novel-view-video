#!/usr/bin/env bash

distil3d_status_usage() {
  cat <<'EOF'
Usage:
  ./distil3d status --profile <profile-directory> <job-name>/<run-id>[/<attempt-id>]
EOF
}

distil3d_logs_usage() {
  cat <<'EOF'
Usage:
  ./distil3d logs --profile <profile-directory> <job-name>/<run-id>[/<attempt-id>] [-f]
EOF
}

distil3d_remove_usage() {
  cat <<'EOF'
Usage:
  ./distil3d remove --profile <profile-directory> <job-name>/<run-id>[/<attempt-id>]
EOF
}

distil3d_parse_attempt_reference() {
  local reference="$1"
  local remainder

  if [[ "${reference}" != */* ]]; then
    printf 'attempt reference must contain job-name/run-id\n' >&2
    return 2
  fi

  DISTIL3D_REFERENCE_JOB_NAME="${reference%%/*}"
  remainder="${reference#*/}"
  DISTIL3D_REFERENCE_RUN_ID="${remainder%%/*}"
  if [[ "${remainder}" == */* ]]; then
    DISTIL3D_REFERENCE_ATTEMPT_ID="${remainder#*/}"
  else
    DISTIL3D_REFERENCE_ATTEMPT_ID=""
  fi

  if [[ -z "${DISTIL3D_REFERENCE_JOB_NAME}" || \
    -z "${DISTIL3D_REFERENCE_RUN_ID}" || \
    "${DISTIL3D_REFERENCE_ATTEMPT_ID}" == */* ]]
  then
    printf 'attempt reference must be job-name/run-id[/attempt-id]\n' >&2
    return 2
  fi
  if [[ "${remainder}" == */* && -z "${DISTIL3D_REFERENCE_ATTEMPT_ID}" ]]; then
    printf 'attempt reference must be job-name/run-id[/attempt-id]\n' >&2
    return 2
  fi
}

distil3d_read_attempt_snapshot() {
  local job_name="$1"
  local run_id="$2"
  local attempt_id="$3"
  local snapshot_output
  local snapshot_line
  local snapshot_key
  local snapshot_value
  local -a attempt_argument=()

  if [[ -n "${attempt_id}" ]]; then
    attempt_argument=(--attempt-id "${attempt_id}")
  fi
  snapshot_output="$(
    docker run \
      --rm \
      --network none \
      --mount "type=bind,src=${DISTIL3D_RUNS_ROOT},dst=/runs,readonly" \
      --entrypoint "" \
      "${DISTIL3D_CORE_IMAGE}" \
      /opt/envs/core/bin/python \
      -m novel_view.cli.main \
      describe-attempt \
      --runs-root /runs \
      --job-name "${job_name}" \
      --run-id "${run_id}" \
      "${attempt_argument[@]}"
  )" || return $?

  DISTIL3D_RESOLVED_ATTEMPT_REF=""
  DISTIL3D_RECORDED_STATE=""
  DISTIL3D_RECORD_PRESENT=""
  while IFS= read -r snapshot_line || [[ -n "${snapshot_line}" ]]; do
    snapshot_key="${snapshot_line%%=*}"
    snapshot_value="${snapshot_line#*=}"
    case "${snapshot_key}" in
      attempt_ref)
        DISTIL3D_RESOLVED_ATTEMPT_REF="${snapshot_value}"
        ;;
      recorded_state)
        DISTIL3D_RECORDED_STATE="${snapshot_value}"
        ;;
      record_present)
        DISTIL3D_RECORD_PRESENT="${snapshot_value}"
        ;;
      *)
        printf 'unknown attempt snapshot field: %s\n' "${snapshot_key}" >&2
        return 2
        ;;
    esac
  done <<< "${snapshot_output}"

  if [[ -z "${DISTIL3D_RESOLVED_ATTEMPT_REF}" || \
    -z "${DISTIL3D_RECORDED_STATE}" || \
    -z "${DISTIL3D_RECORD_PRESENT}" ]]
  then
    printf 'attempt snapshot is incomplete\n' >&2
    return 2
  fi
}

distil3d_find_exact_container() {
  local attempt_ref="$1"
  local container_output
  local container_id
  local -a container_ids=()

  container_output="$(
    docker ps \
      --all \
      --filter "label=io.distil3d.attempt-ref=${attempt_ref}" \
      --format '{{.ID}}'
  )" || return $?
  while IFS= read -r container_id; do
    if [[ -n "${container_id}" ]]; then
      container_ids[${#container_ids[@]}]="${container_id}"
    fi
  done <<< "${container_output}"

  if [[ ${#container_ids[@]} -gt 1 ]]; then
    printf 'multiple containers have exact attempt reference: %s\n' \
      "${attempt_ref}" >&2
    return 2
  fi
  DISTIL3D_RESOLVED_CONTAINER_ID="${container_ids[0]:-}"
}

distil3d_resolve_attempt() {
  local reference="$1"
  local active_output
  local active_container_id
  local active_attempt_ref
  local extra
  local resolved_attempt_id
  local -a active_container_ids=()
  local -a active_attempt_refs=()

  distil3d_parse_attempt_reference "${reference}"
  if [[ -n "${DISTIL3D_REFERENCE_ATTEMPT_ID}" ]]; then
    DISTIL3D_RESOLVED_ATTEMPT_REF="${reference}"
    distil3d_find_exact_container "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
    distil3d_read_attempt_snapshot \
      "${DISTIL3D_REFERENCE_JOB_NAME}" \
      "${DISTIL3D_REFERENCE_RUN_ID}" \
      "${DISTIL3D_REFERENCE_ATTEMPT_ID}"
  else
    active_output="$(
      docker ps \
        --filter "label=io.distil3d.job=${DISTIL3D_REFERENCE_JOB_NAME}" \
        --filter "label=io.distil3d.run-id=${DISTIL3D_REFERENCE_RUN_ID}" \
        --format '{{.ID}} {{.Label "io.distil3d.attempt-ref"}}'
    )" || return $?
    while read -r active_container_id active_attempt_ref extra; do
      if [[ -n "${active_container_id}" ]]; then
        active_container_ids[${#active_container_ids[@]}]="${active_container_id}"
        active_attempt_refs[${#active_attempt_refs[@]}]="${active_attempt_ref}"
      fi
    done <<< "${active_output}"

    if [[ ${#active_container_ids[@]} -gt 1 ]]; then
      printf 'ambiguous run reference: %s/%s\n' \
        "${DISTIL3D_REFERENCE_JOB_NAME}" \
        "${DISTIL3D_REFERENCE_RUN_ID}" >&2
      for active_attempt_ref in "${active_attempt_refs[@]}"; do
        printf 'candidate=%s\n' "${active_attempt_ref}" >&2
      done
      return 2
    fi

    if [[ ${#active_container_ids[@]} -eq 1 ]]; then
      DISTIL3D_RESOLVED_CONTAINER_ID="${active_container_ids[0]}"
      DISTIL3D_RESOLVED_ATTEMPT_REF="${active_attempt_refs[0]}"
      resolved_attempt_id="${DISTIL3D_RESOLVED_ATTEMPT_REF##*/}"
      distil3d_read_attempt_snapshot \
        "${DISTIL3D_REFERENCE_JOB_NAME}" \
        "${DISTIL3D_REFERENCE_RUN_ID}" \
        "${resolved_attempt_id}"
      if [[ "${DISTIL3D_RECORD_PRESENT}" != "true" ]]; then
        printf 'container without attempt record requires exact reference: %s\n' \
          "${DISTIL3D_RESOLVED_ATTEMPT_REF}" >&2
        return 2
      fi
    else
      distil3d_read_attempt_snapshot \
        "${DISTIL3D_REFERENCE_JOB_NAME}" \
        "${DISTIL3D_REFERENCE_RUN_ID}" \
        ""
      distil3d_find_exact_container "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
    fi
  fi

  if [[ -z "${DISTIL3D_RESOLVED_CONTAINER_ID}" && \
    "${DISTIL3D_RECORD_PRESENT}" != "true" ]]
  then
    printf 'attempt not found: %s\n' "${DISTIL3D_RESOLVED_ATTEMPT_REF}" >&2
    return 2
  fi
}

distil3d_inspect_container() {
  local container_id="$1"
  local container_output
  local extra

  container_output="$(
    docker inspect \
      --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}' \
      "${container_id}"
  )" || return $?
  read -r \
    DISTIL3D_CONTAINER_STATE \
    DISTIL3D_CONTAINER_EXIT_CODE \
    DISTIL3D_CONTAINER_OOM_KILLED \
    extra <<< "${container_output}"
  case "${DISTIL3D_CONTAINER_STATE}" in
    created | running | paused | restarting)
      DISTIL3D_CONTAINER_EXIT_CODE="null"
      ;;
  esac
}

distil3d_inspect_resolved_container() {
  if [[ -z "${DISTIL3D_RESOLVED_CONTAINER_ID}" ]]; then
    DISTIL3D_CONTAINER_STATE="absent"
    DISTIL3D_CONTAINER_EXIT_CODE="null"
    DISTIL3D_CONTAINER_OOM_KILLED="null"
  else
    distil3d_inspect_container "${DISTIL3D_RESOLVED_CONTAINER_ID}"
  fi
}

distil3d_print_container_state() {
  printf 'container_state=%s\n' "${DISTIL3D_CONTAINER_STATE}"
  printf 'container_exit_code=%s\n' "${DISTIL3D_CONTAINER_EXIT_CODE}"
  printf 'container_oom_killed=%s\n' "${DISTIL3D_CONTAINER_OOM_KILLED}"
}

distil3d_print_status() {
  distil3d_inspect_resolved_container

  printf 'recorded_state=%s\n' "${DISTIL3D_RECORDED_STATE}"
  distil3d_print_container_state
  printf 'attempt_ref=%s\n' "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
}

distil3d_status_main() {
  local profile_argument=""
  local reference=""

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
        distil3d_status_usage
        return 0
        ;;
      --*)
        printf 'unknown status argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${reference}" ]]; then
          printf 'status accepts exactly one attempt reference\n' >&2
          return 2
        fi
        reference="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" || -z "${reference}" ]]; then
    distil3d_status_usage >&2
    return 2
  fi
  distil3d_load_profile "${profile_argument}"
  distil3d_resolve_attempt "${reference}"
  distil3d_print_status
}

distil3d_logs_main() {
  local profile_argument=""
  local reference=""
  local follow=false
  local -a follow_argument=()

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
      -f | --follow)
        follow=true
        shift
        ;;
      -h | --help)
        distil3d_logs_usage
        return 0
        ;;
      --*)
        printf 'unknown logs argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${reference}" ]]; then
          printf 'logs accepts exactly one attempt reference\n' >&2
          return 2
        fi
        reference="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" || -z "${reference}" ]]; then
    distil3d_logs_usage >&2
    return 2
  fi
  distil3d_load_profile "${profile_argument}"
  distil3d_resolve_attempt "${reference}"
  if [[ -z "${DISTIL3D_RESOLVED_CONTAINER_ID}" ]]; then
    printf 'container logs are unavailable: %s\n' \
      "${DISTIL3D_RESOLVED_ATTEMPT_REF}" >&2
    return 1
  fi
  if [[ "${follow}" == "true" ]]; then
    follow_argument=(--follow)
  fi
  docker logs "${follow_argument[@]}" "${DISTIL3D_RESOLVED_CONTAINER_ID}"
}

distil3d_remove_main() {
  local profile_argument=""
  local reference=""

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
        distil3d_remove_usage
        return 0
        ;;
      --*)
        printf 'unknown remove argument: %s\n' "$1" >&2
        return 2
        ;;
      *)
        if [[ -n "${reference}" ]]; then
          printf 'remove accepts exactly one attempt reference\n' >&2
          return 2
        fi
        reference="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${profile_argument}" || -z "${reference}" ]]; then
    distil3d_remove_usage >&2
    return 2
  fi

  distil3d_load_profile "${profile_argument}"
  distil3d_resolve_attempt "${reference}"
  if [[ -z "${DISTIL3D_RESOLVED_CONTAINER_ID}" ]]; then
    printf 'container already absent: %s\n' \
      "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
    return 0
  fi

  distil3d_inspect_resolved_container
  case "${DISTIL3D_CONTAINER_STATE}" in
    exited | dead)
      docker rm "${DISTIL3D_RESOLVED_CONTAINER_ID}" >/dev/null
      printf 'removed=%s\n' "${DISTIL3D_RESOLVED_ATTEMPT_REF}"
      ;;
    running | paused | restarting)
      printf 'container is running; stop it first: %s\n' \
        "${DISTIL3D_RESOLVED_ATTEMPT_REF}" >&2
      return 2
      ;;
    *)
      printf 'container is not finished: %s state=%s\n' \
        "${DISTIL3D_RESOLVED_ATTEMPT_REF}" \
        "${DISTIL3D_CONTAINER_STATE}" >&2
      return 2
      ;;
  esac
}
