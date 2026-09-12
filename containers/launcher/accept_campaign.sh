#!/usr/bin/env bash

_distil3d_accept_bind_attempt() {
  local file="$1"
  local sentinel="$2"
  local attempt_ref="$3"
  local locator

  locator="$(_distil3d_accept_attempt_locator "${attempt_ref}")" || return $?
  _distil3d_accept_replace "${file}" "${sentinel}" "${locator}"
}

_distil3d_accept_bind_euvs_source() {
  local ref="$1"
  local name
  local file

  for name in euvs-pair-autoregressive-v1 euvs-pair-autoregressive-cp2 \
    euvs-pair-source-reseed-v1
  do
    file="${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml"
    _distil3d_accept_bind_attempt "${file}" \
      euvs-source-views-v1/replace-run/attempts/replace-attempt "${ref}" \
      || return $?
  done
}

_distil3d_accept_bind_euvs_evaluation() {
  _distil3d_accept_bind_attempt \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/euvs-evaluation-run-v1/run.yaml" \
    euvs-pair-autoregressive-v1/replace-run/attempts/replace-attempt "$1"
}

_distil3d_accept_bind_euvs_comparison() {
  _distil3d_accept_bind_attempt \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/euvs-base-tuned-comparison-v1/run.yaml" \
    euvs-evaluation-run-v1/replace-run/attempts/replace-attempt "$1"
}

_distil3d_accept_bind_gaussian_handoff() {
  _distil3d_accept_bind_attempt \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-png-handoff-v1/run.yaml" \
    gaussian-dense-independent-v1/replace-run/attempts/replace-attempt "$1" \
    || return $?
  _distil3d_accept_bind_attempt \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-png-handoff-v1/run.yaml" \
    gaussian-dense-overlap21-v1/replace-run/attempts/replace-attempt "$2"
}

_distil3d_accept_bind_selection() {
  local version="$1"
  local training_ref="$2"
  local name="r4c-lora-edm"

  if [[ "${version}" == 2 ]]; then name=r4c-lora-lidar-depth; fi
  _distil3d_accept_bind_attempt \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/r4c-select-v${version}/run.yaml" \
    "${name}/replace-run/attempts/replace-attempt" "${training_ref}"
}

_distil3d_accept_bind_ddw_evaluation() {
  local file="${DISTIL3D_ACCEPT_JOB_ROOT}/r4c-ddw-evaluation/run.yaml"

  _distil3d_accept_replace "${file}" \
    r4c-lora-edm/replace-run/attempts/replace-attempt/checkpoints/step-000000154.json \
    "${DISTIL3D_ACCEPT_V1_SELECTED_RECORD}" || return $?
  _distil3d_accept_replace "${file}" \
    r4c-lora-lidar-depth/replace-run/attempts/replace-attempt/checkpoints/step-000000154.json \
    "${DISTIL3D_ACCEPT_V2_SELECTED_RECORD}"
}

_distil3d_accept_has_result() {
  awk -F '\t' -v wanted="$1" 'NR > 1 && $1 == wanted {found=1} END {exit !found}' \
    "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/summary.tsv"
}

_distil3d_accept_close_unstarted() {
  local reason="$1"
  local case_id
  local phase
  local required

  while IFS='|' read -r case_id phase required; do
    if ! _distil3d_accept_has_result "${case_id}"; then
      _distil3d_accept_skip "${case_id}" "${phase}" "${required}" "${reason}"
    fi
  done <<'EOF'
ac:scenario:sc:euvs-geometry-campaign-v1|inference|true
ac:scenario:sc:euvs-pair-autoregressive-v1|inference|true
ac:scenario:sc:euvs-pair-autoregressive-v1:cp2|inference|true
ac:scenario:sc:euvs-pair-source-reseed-v1|inference|true
ac:scenario:sc:euvs-evaluation-run-v1|evaluation|true
ac:scenario:sc:euvs-base-tuned-comparison-v1|comparison|true
ac:scenario:sc:gaussian-full-sequence-v1|inference|true
ac:scenario:sc:gaussian-independent-clip-v1|inference|true
ac:scenario:sc:gaussian-dense-independent-v1|inference|true
ac:scenario:sc:gaussian-dense-overlap21-v1|inference|true
ac:scenario:sc:gaussian-dense-png-handoff-v1|handoff|true
ac:scenario:sc:ddw-preparation-v1|preparation|true
ac:test:tc:model:test-gen3c-activation-checkpointing:gen3c-activation-checkpointing-tests|native-api|true
ac:scenario:sc:gen3c-training-fresh-v1|supported-v1|true
ac:scenario:sc:gen3c-training-resume-v1|supported-v1|true
ac:scenario:sc:gen3c-checkpoint-selection-v1|supported-v1|true
ac:scenario:sc:gen3c-training-fresh-v2|experimental-v2|true
ac:scenario:sc:gen3c-training-resume-v2|experimental-v2|true
ac:scenario:sc:gen3c-checkpoint-selection-v2|experimental-v2|true
ac:scenario:sc:gen3c-ddw-evaluation-v2|matched-evaluation|true
ac:scenario:sc:waymo-depth-candidates-v2|legacy-waymo|true
ac:scenario:sc:waymo-depth-selection-gate-v2|legacy-waymo|true
ac:scenario:sc:waymo-ddw-engineering-canary-v1|legacy-waymo|true
ac:scenario:sc:waymo-ddw-v2-canary-v1|legacy-waymo|true
ac:scenario:sc:waymo-ddw-v3-probe-v1|legacy-waymo|true
ac:scenario:sc:waymo-ddw-survey-v1|legacy-waymo|true
EOF
}

_distil3d_accept_closure() {
  local status=succeeded
  local note='all required cases succeeded'
  local actual_core
  local actual_moge

  if actual_core="$(docker image inspect --format '{{.Id}}' "${DISTIL3D_CORE_IMAGE}")" \
    && actual_moge="$(docker image inspect --format '{{.Id}}' "${DISTIL3D_MOGE_IMAGE}")" \
    && [[ "${actual_core}" == "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}" ]] \
    && [[ "${actual_moge}" == "${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" ]] \
    && distil3d_check_candidate_image "${actual_core}" core \
      "${DISTIL3D_ACCEPT_SOURCE_REVISION}" "${DISTIL3D_ACCEPT_LOCK_REVISION}" \
    && distil3d_check_candidate_image "${actual_moge}" moge \
      "${DISTIL3D_ACCEPT_SOURCE_REVISION}" "${DISTIL3D_ACCEPT_LOCK_REVISION}"
  then
    :
  else
    DISTIL3D_ACCEPT_REQUIRED_FAILED=1
    status=failed
    note='candidate image identity changed before closure'
  fi
  if [[ ${DISTIL3D_ACCEPT_REQUIRED_FAILED} -ne 0 || \
    ${DISTIL3D_ACCEPT_INTERRUPTED} -ne 0 ]]
  then
    status=failed
    note='one or more required cases did not succeed'
  fi
  _distil3d_accept_write_no_attempt closure closure true "${status}" \
    "$([[ "${status}" == succeeded ]] && printf 0 || printf 1)" "${note}"
  [[ "${status}" == succeeded ]]
}

_distil3d_accept_stop_after_abort() {
  if [[ ${DISTIL3D_ACCEPT_ABORT} -eq 0 ]]; then return 0; fi
  _distil3d_accept_close_unstarted 'skipped:campaign-interrupted'
  _distil3d_accept_closure || true
  return 130
}

_distil3d_accept_waymo_case() {
  local case_id="$1"
  local name="$2"
  local status

  _distil3d_accept_plan_case "${case_id}" legacy-waymo true "${name}" || return $?
  if _distil3d_accept_doctor_job "${name}" \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/doctor.log" 2>&1
  then
    _distil3d_accept_run_case "${case_id}" legacy-waymo true "${name}"
  else
    status=$?
    _distil3d_accept_write_no_attempt "${case_id}" legacy-waymo true \
      failed "${status}" 'historical Waymo job doctor failed'
    return "${status}"
  fi
}

_distil3d_accept_waymo_campaign() {
  local gate_ok=false
  local name

  DISTIL3D_ACCEPT_DEPTH_SELECTION=''
  _distil3d_accept_waymo_case ac:scenario:sc:waymo-depth-candidates-v2 \
    waymo-depth-candidates || true
  _distil3d_accept_stop_after_abort || return $?
  if _distil3d_accept_waymo_case ac:scenario:sc:waymo-depth-selection-gate-v2 \
    waymo-depth-selection-v2
  then
    if [[ "${DISTIL3D_ACCEPT_WAYMO_DDW_ALLOWED}" == true ]]; then
      gate_ok=true
      DISTIL3D_ACCEPT_DEPTH_SELECTION="$(_distil3d_accept_attempt_locator \
        "${DISTIL3D_ACCEPT_ACTIVE_REF}")/depth-selection.json"
      for name in waymo-ddw-canary-v1 waymo-ddw-canary-v2 waymo-ddw-probe waymo-ddw-survey; do
        _distil3d_accept_bind_attempt "${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml" \
          waymo-depth-selection-v2/example/attempts/0001 \
          "${DISTIL3D_ACCEPT_ACTIVE_REF}" || return $?
      done
    fi
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${gate_ok}" == true ]]; then
    _distil3d_accept_waymo_case ac:scenario:sc:waymo-ddw-engineering-canary-v1 \
      waymo-ddw-canary-v1 || true
    _distil3d_accept_stop_after_abort || return $?
    _distil3d_accept_waymo_case ac:scenario:sc:waymo-ddw-v2-canary-v1 \
      waymo-ddw-canary-v2 || true
    _distil3d_accept_stop_after_abort || return $?
    _distil3d_accept_waymo_case ac:scenario:sc:waymo-ddw-v3-probe-v1 \
      waymo-ddw-probe || true
    _distil3d_accept_stop_after_abort || return $?
    _distil3d_accept_waymo_case ac:scenario:sc:waymo-ddw-survey-v1 \
      waymo-ddw-survey || true
    _distil3d_accept_stop_after_abort || return $?
  else
    _distil3d_accept_skip ac:scenario:sc:waymo-ddw-engineering-canary-v1 \
      legacy-waymo true 'skipped:depth-selection-did-not-admit-moge'
    _distil3d_accept_skip ac:scenario:sc:waymo-ddw-v2-canary-v1 \
      legacy-waymo true 'skipped:depth-selection-did-not-admit-moge'
    _distil3d_accept_skip ac:scenario:sc:waymo-ddw-v3-probe-v1 \
      legacy-waymo true 'skipped:depth-selection-did-not-admit-moge'
    _distil3d_accept_skip ac:scenario:sc:waymo-ddw-survey-v1 \
      legacy-waymo true 'skipped:depth-selection-did-not-admit-moge'
  fi
}

_distil3d_accept_campaign() {
  local source_planned=false
  local gaussian_full_planned=false
  local gaussian_clip_planned=false
  local gaussian_independent_planned=false
  local gaussian_overlap_planned=false
  local preparation_planned=false
  local v1_planned=false
  local v2_planned=false
  local source_ok=false
  local cp1_ok=false
  local evaluation_ok=false
  local gaussian_independent_ok=false
  local gaussian_overlap_ok=false
  local preparation_ok=false
  local te_recompute_ok=false
  local v1_fresh_ok=false
  local v1_resume_ok=false
  local v1_selection_ok=false
  local v2_fresh_ok=false
  local v2_resume_ok=false
  local v2_selection_ok=false
  local source_ref=''
  local cp1_ref=''
  local evaluation_ref=''
  local gaussian_independent_ref=''
  local gaussian_overlap_ref=''
  local v1_fresh_ref=''
  local v2_fresh_ref=''

  if _distil3d_accept_plan_case \
    ac:scenario:sc:euvs-geometry-campaign-v1 inference true euvs-source-views-v1
  then source_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gaussian-full-sequence-v1 inference true gaussian-full-sequence-v1
  then gaussian_full_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gaussian-independent-clip-v1 inference true gaussian-independent-clip-v1
  then gaussian_clip_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gaussian-dense-independent-v1 inference true gaussian-dense-independent-v1
  then gaussian_independent_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gaussian-dense-overlap21-v1 inference true gaussian-dense-overlap21-v1
  then gaussian_overlap_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:ddw-preparation-v1 preparation true stage11-waymo-one-item
  then preparation_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gen3c-training-fresh-v1 supported-v1 true r4c-lora-edm
  then v1_planned=true; fi
  if _distil3d_accept_plan_case \
    ac:scenario:sc:gen3c-training-fresh-v2 experimental-v2 true r4c-lora-lidar-depth
  then v2_planned=true; fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${source_planned}" == true ]] && _distil3d_accept_run_case \
    ac:scenario:sc:euvs-geometry-campaign-v1 inference true euvs-source-views-v1
  then
    source_ok=true
    source_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${source_ok}" == true ]]; then
    if _distil3d_accept_bind_euvs_source "${source_ref}"; then :; else
      DISTIL3D_ACCEPT_ABORT=1
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:euvs-pair-autoregressive-v1 \
      inference true 'skipped:first-failed(euvs-source-views)'
    _distil3d_accept_skip ac:scenario:sc:euvs-pair-autoregressive-v1:cp2 \
      inference true 'skipped:first-failed(euvs-source-views)'
    _distil3d_accept_skip ac:scenario:sc:euvs-pair-source-reseed-v1 \
      inference true 'skipped:first-failed(euvs-source-views)'
    _distil3d_accept_skip ac:scenario:sc:euvs-evaluation-run-v1 \
      evaluation true 'skipped:first-failed(euvs-generation)'
    _distil3d_accept_skip ac:scenario:sc:euvs-base-tuned-comparison-v1 \
      comparison true 'skipped:first-failed(euvs-evaluation)'
  fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${source_ok}" == true ]]; then
    if _distil3d_accept_plan_case ac:scenario:sc:euvs-pair-autoregressive-v1 \
      inference true euvs-pair-autoregressive-v1
    then
      if _distil3d_accept_run_case ac:scenario:sc:euvs-pair-autoregressive-v1 \
        inference true euvs-pair-autoregressive-v1
      then
        cp1_ok=true
        cp1_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
      fi
    fi
    _distil3d_accept_stop_after_abort || return $?
    if [[ "${cp1_ok}" == true ]]; then
      if _distil3d_accept_plan_case ac:scenario:sc:euvs-pair-autoregressive-v1:cp2 \
        inference true euvs-pair-autoregressive-cp2
      then
        _distil3d_accept_run_case ac:scenario:sc:euvs-pair-autoregressive-v1:cp2 \
          inference true euvs-pair-autoregressive-cp2 || true
      fi
    else
      _distil3d_accept_skip ac:scenario:sc:euvs-pair-autoregressive-v1:cp2 \
        inference true 'skipped:first-failed(euvs-cp1)'
    fi
    _distil3d_accept_stop_after_abort || return $?
    if _distil3d_accept_plan_case ac:scenario:sc:euvs-pair-source-reseed-v1 \
      inference true euvs-pair-source-reseed-v1
    then
      _distil3d_accept_run_case ac:scenario:sc:euvs-pair-source-reseed-v1 \
        inference true euvs-pair-source-reseed-v1 || true
    fi
    _distil3d_accept_stop_after_abort || return $?
    if [[ "${cp1_ok}" == true ]]; then
      _distil3d_accept_bind_euvs_evaluation "${cp1_ref}" || return $?
      if _distil3d_accept_plan_case ac:scenario:sc:euvs-evaluation-run-v1 \
        evaluation true euvs-evaluation-run-v1
      then
        if _distil3d_accept_run_case ac:scenario:sc:euvs-evaluation-run-v1 \
          evaluation true euvs-evaluation-run-v1
        then
          evaluation_ok=true
          evaluation_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
        fi
      fi
    else
      _distil3d_accept_skip ac:scenario:sc:euvs-evaluation-run-v1 evaluation \
        true 'skipped:first-failed(euvs-cp1)'
    fi
    _distil3d_accept_stop_after_abort || return $?
    if [[ "${evaluation_ok}" == true ]]; then
      _distil3d_accept_bind_euvs_comparison "${evaluation_ref}" || return $?
      if _distil3d_accept_plan_case ac:scenario:sc:euvs-base-tuned-comparison-v1 \
        comparison true euvs-base-tuned-comparison-v1
      then
        if _distil3d_accept_doctor_job euvs-base-tuned-comparison-v1 \
          >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/doctor.log" 2>&1
        then
          _distil3d_accept_run_case ac:scenario:sc:euvs-base-tuned-comparison-v1 \
            comparison true euvs-base-tuned-comparison-v1 || true
        else
          _distil3d_accept_skip ac:scenario:sc:euvs-base-tuned-comparison-v1 \
            comparison true 'skipped:missing-or-invalid-external-reference'
        fi
      fi
    else
      _distil3d_accept_skip ac:scenario:sc:euvs-base-tuned-comparison-v1 \
        comparison true 'skipped:first-failed(euvs-evaluation)'
    fi
  fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${gaussian_full_planned}" == true ]]; then
    _distil3d_accept_run_case ac:scenario:sc:gaussian-full-sequence-v1 \
      inference true gaussian-full-sequence-v1 || true
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${gaussian_clip_planned}" == true ]]; then
    _distil3d_accept_run_case ac:scenario:sc:gaussian-independent-clip-v1 \
      inference true gaussian-independent-clip-v1 || true
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${gaussian_independent_planned}" == true ]] && \
    _distil3d_accept_run_case ac:scenario:sc:gaussian-dense-independent-v1 \
      inference true gaussian-dense-independent-v1
  then
    gaussian_independent_ok=true
    gaussian_independent_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${gaussian_overlap_planned}" == true ]] && \
    _distil3d_accept_run_case ac:scenario:sc:gaussian-dense-overlap21-v1 \
      inference true gaussian-dense-overlap21-v1
  then
    gaussian_overlap_ok=true
    gaussian_overlap_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${gaussian_independent_ok}" == true && \
    "${gaussian_overlap_ok}" == true ]]
  then
    _distil3d_accept_bind_gaussian_handoff "${gaussian_independent_ref}" \
      "${gaussian_overlap_ref}" || return $?
    if _distil3d_accept_plan_case ac:scenario:sc:gaussian-dense-png-handoff-v1 \
      handoff true gaussian-dense-png-handoff-v1
    then
      _distil3d_accept_run_case ac:scenario:sc:gaussian-dense-png-handoff-v1 \
        handoff true gaussian-dense-png-handoff-v1 || true
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gaussian-dense-png-handoff-v1 handoff \
      true 'skipped:first-failed(gaussian-v3-or-v4)'
  fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${preparation_planned}" == true ]] && _distil3d_accept_run_case \
    ac:scenario:sc:ddw-preparation-v1 preparation true stage11-waymo-one-item
  then preparation_ok=true; fi
  _distil3d_accept_stop_after_abort || return $?

  if _distil3d_accept_te_recompute; then te_recompute_ok=true; fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${preparation_ok}" == true && "${te_recompute_ok}" == true && \
    "${v1_planned}" == true ]] && \
    _distil3d_accept_run_case ac:scenario:sc:gen3c-training-fresh-v1 \
      supported-v1 true r4c-lora-edm
  then
    v1_fresh_ok=true
    v1_fresh_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
  elif [[ ( "${preparation_ok}" != true || "${te_recompute_ok}" != true ) && \
    "${v1_planned}" == true ]]; then
    _distil3d_accept_skip ac:scenario:sc:gen3c-training-fresh-v1 supported-v1 \
      true 'skipped:first-failed(ddw-preparation-or-te-recompute)'
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${v1_fresh_ok}" == true ]]; then
    if _distil3d_accept_read_checkpoint 1 "${v1_fresh_ref}"; then
      if _distil3d_accept_resume_case ac:scenario:sc:gen3c-training-resume-v1 \
        supported-v1 true r4c-lora-edm "${v1_fresh_ref}" \
        "${DISTIL3D_ACCEPT_CHECKPOINT}"
      then v1_resume_ok=true; fi
    else
      DISTIL3D_ACCEPT_REQUIRED_FAILED=1
      _distil3d_accept_skip ac:scenario:sc:gen3c-training-resume-v1 supported-v1 \
        true 'skipped:fresh-checkpoint-record-unreadable'
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gen3c-training-resume-v1 supported-v1 \
      true 'skipped:first-failed(v1-fresh)'
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${v1_fresh_ok}" == true && "${v1_resume_ok}" == true ]]; then
    _distil3d_accept_bind_selection 1 "${v1_fresh_ref}" || return $?
    if _distil3d_accept_plan_case ac:scenario:sc:gen3c-checkpoint-selection-v1 \
      supported-v1 true r4c-select-v1
    then
      if _distil3d_accept_run_case ac:scenario:sc:gen3c-checkpoint-selection-v1 \
        supported-v1 true r4c-select-v1 1 "${v1_fresh_ref}"
      then
        v1_selection_ok=true
        DISTIL3D_ACCEPT_V1_SELECTED_RECORD="${DISTIL3D_ACCEPT_SELECTED_RECORD}"
        DISTIL3D_ACCEPT_V1_SELECTED_STEP="${DISTIL3D_ACCEPT_SELECTED_STEP}"
      fi
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gen3c-checkpoint-selection-v1 supported-v1 \
      true 'skipped:first-failed(v1-fresh-or-resume)'
  fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${preparation_ok}" == true && "${te_recompute_ok}" == true && \
    "${v2_planned}" == true ]] && \
    _distil3d_accept_run_case ac:scenario:sc:gen3c-training-fresh-v2 \
      experimental-v2 true r4c-lora-lidar-depth
  then
    v2_fresh_ok=true
    v2_fresh_ref="${DISTIL3D_ACCEPT_ACTIVE_REF}"
  elif [[ ( "${preparation_ok}" != true || "${te_recompute_ok}" != true ) && \
    "${v2_planned}" == true ]]; then
    _distil3d_accept_skip ac:scenario:sc:gen3c-training-fresh-v2 experimental-v2 \
      true 'skipped:first-failed(ddw-preparation-or-te-recompute)'
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${v2_fresh_ok}" == true ]]; then
    if _distil3d_accept_read_checkpoint 2 "${v2_fresh_ref}"; then
      if _distil3d_accept_resume_case ac:scenario:sc:gen3c-training-resume-v2 \
        experimental-v2 true r4c-lora-lidar-depth "${v2_fresh_ref}" \
        "${DISTIL3D_ACCEPT_CHECKPOINT}"
      then v2_resume_ok=true; fi
    else
      DISTIL3D_ACCEPT_REQUIRED_FAILED=1
      _distil3d_accept_skip ac:scenario:sc:gen3c-training-resume-v2 experimental-v2 \
        true 'skipped:fresh-checkpoint-record-unreadable'
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gen3c-training-resume-v2 experimental-v2 \
      true 'skipped:first-failed(v2-fresh)'
  fi
  _distil3d_accept_stop_after_abort || return $?
  if [[ "${v2_fresh_ok}" == true && "${v2_resume_ok}" == true ]]; then
    _distil3d_accept_bind_selection 2 "${v2_fresh_ref}" || return $?
    if _distil3d_accept_plan_case ac:scenario:sc:gen3c-checkpoint-selection-v2 \
      experimental-v2 true r4c-select-v2
    then
      if _distil3d_accept_run_case ac:scenario:sc:gen3c-checkpoint-selection-v2 \
        experimental-v2 true r4c-select-v2 2 "${v2_fresh_ref}"
      then
        v2_selection_ok=true
        DISTIL3D_ACCEPT_V2_SELECTED_RECORD="${DISTIL3D_ACCEPT_SELECTED_RECORD}"
        DISTIL3D_ACCEPT_V2_SELECTED_STEP="${DISTIL3D_ACCEPT_SELECTED_STEP}"
      fi
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gen3c-checkpoint-selection-v2 \
      experimental-v2 true 'skipped:first-failed(v2-fresh-or-resume)'
  fi
  _distil3d_accept_stop_after_abort || return $?

  if [[ "${preparation_ok}" == true && "${v1_selection_ok}" == true && \
    "${v2_selection_ok}" == true && \
    "${DISTIL3D_ACCEPT_V1_SELECTED_STEP}" == "${DISTIL3D_ACCEPT_V2_SELECTED_STEP}" ]]
  then
    _distil3d_accept_bind_ddw_evaluation || return $?
    if _distil3d_accept_plan_case ac:scenario:sc:gen3c-ddw-evaluation-v2 \
      matched-evaluation true r4c-ddw-evaluation
    then
      _distil3d_accept_run_case ac:scenario:sc:gen3c-ddw-evaluation-v2 \
        matched-evaluation true r4c-ddw-evaluation || true
    fi
  else
    _distil3d_accept_skip ac:scenario:sc:gen3c-ddw-evaluation-v2 \
      matched-evaluation true 'skipped:missing-or-unmatched-v1-v2-selection'
  fi
  _distil3d_accept_stop_after_abort || return $?

  _distil3d_accept_waymo_campaign || return $?
  _distil3d_accept_closure
}
