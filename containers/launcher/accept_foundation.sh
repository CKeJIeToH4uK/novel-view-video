#!/usr/bin/env bash

_distil3d_accept_copy_job() {
  local source="$1"
  local name="$2"
  local destination="${DISTIL3D_ACCEPT_JOB_ROOT}/${name}"

  mkdir -p -- "${destination}"
  cp -- "${DISTIL3D_REPOSITORY_ROOT}/${source}/run.yaml" \
    "${destination}/run.yaml"
}

_distil3d_accept_replace() {
  local file="$1"
  local old="$2"
  local new="$3"

  if ! grep -Fq -- "${old}" "${file}"; then
    printf 'acceptance form does not contain its fixed sentinel: %s\n' \
      "${file}" >&2
    return 2
  fi
  sed -i.bak "s|${old}|${new}|g" "${file}" || return $?
  rm -- "${file}.bak"
}

_distil3d_accept_prepare_jobs() {
  local name
  local file

  DISTIL3D_ACCEPT_CAMPAIGN_KEY="${DISTIL3D_ACCEPT_INVOCATION_ID}-${DISTIL3D_ACCEPT_CANDIDATE_ID}"
  DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT="jobs/local/${DISTIL3D_ACCEPT_CAMPAIGN_KEY}"
  DISTIL3D_ACCEPT_JOB_ROOT="${DISTIL3D_REPOSITORY_ROOT}/${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}"
  mkdir -p -- "${DISTIL3D_ACCEPT_JOB_ROOT}"

  for name in euvs-source-views-v1 euvs-pair-autoregressive-v1 \
    euvs-pair-autoregressive-cp2 euvs-pair-source-reseed-v1 \
    euvs-evaluation-run-v1 euvs-base-tuned-comparison-v1 \
    gaussian-full-sequence-v1 gaussian-independent-clip-v1 \
    gaussian-dense-independent-v1 gaussian-dense-overlap21-v1 \
    gaussian-dense-png-handoff-v1
  do
    _distil3d_accept_copy_job "jobs/acceptance/${name}" "${name}" || return $?
  done
  _distil3d_accept_copy_job jobs/waymo/r4c-ddw-prepared \
    stage11-waymo-one-item || return $?
  _distil3d_accept_copy_job jobs/waymo/r4c-lora-edm r4c-lora-edm || return $?
  _distil3d_accept_copy_job jobs/waymo/r4c-lora-lidar-depth r4c-lora-lidar-depth || return $?
  _distil3d_accept_copy_job jobs/examples/r4c-select-v1 r4c-select-v1 || return $?
  _distil3d_accept_copy_job jobs/examples/r4c-select-v2 r4c-select-v2 || return $?
  _distil3d_accept_copy_job jobs/examples/r4c-ddw-evaluation r4c-ddw-evaluation || return $?
  for name in depth-candidates depth-selection-v2 ddw-canary-v1 ddw-canary-v2 \
    ddw-probe ddw-survey
  do
    _distil3d_accept_copy_job "jobs/legacy/waymo/${name}" "waymo-${name}" || return $?
  done
  for name in waymo-depth-candidates waymo-depth-selection-v2 waymo-ddw-canary-v1 \
    waymo-ddw-canary-v2 waymo-ddw-probe waymo-ddw-survey
  do
    _distil3d_accept_replace "${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml" \
      'selection: selections/local/' \
      'selection: selections/local/stage11/legacy/' || return $?
  done

  for name in euvs-source-views-v1 euvs-pair-autoregressive-v1 \
    euvs-pair-autoregressive-cp2 euvs-pair-source-reseed-v1 \
    euvs-evaluation-run-v1 euvs-base-tuned-comparison-v1
  do
    file="${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml"
    _distil3d_accept_replace "${file}" selections/euvs/one-pair.yaml \
      selections/local/stage11/euvs-one-pair.yaml || return $?
  done
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/euvs-base-tuned-comparison-v1/run.yaml" \
    external-euvs-tuned-evaluation/replace-run/attempts/replace-attempt \
    external-euvs-tuned-evaluation/reference/attempts/accepted || return $?

  for name in gaussian-full-sequence-v1 gaussian-independent-clip-v1 \
    gaussian-dense-independent-v1 gaussian-dense-overlap21-v1
  do
    file="${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml"
    _distil3d_accept_replace "${file}" gaussian/replace-export \
      gaussian/stage11 || return $?
  done
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-independent-v1/run.yaml" \
    gaussian/replace-camera-table gaussian/stage11 || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-overlap21-v1/run.yaml" \
    gaussian/replace-camera-table gaussian/stage11 || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-full-sequence-v1/run.yaml" \
    selections/gaussian/full-sequence.yaml \
    selections/local/stage11/gaussian-full-sequence.yaml || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-independent-clip-v1/run.yaml" \
    selections/gaussian/independent-clip.yaml \
    selections/local/stage11/gaussian-independent-clip.yaml || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-independent-v1/run.yaml" \
    selections/gaussian/dense-independent.yaml \
    selections/local/stage11/gaussian-dense-independent.yaml || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-overlap21-v1/run.yaml" \
    selections/gaussian/dense-overlap21.yaml \
    selections/local/stage11/gaussian-dense-overlap21.yaml || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/gaussian-dense-png-handoff-v1/run.yaml" \
    selections/gaussian/dense-handoff.yaml \
    selections/local/stage11/gaussian-dense-handoff.yaml || return $?

  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/stage11-waymo-one-item/run.yaml" \
    'name: r4c-ddw-prepared' 'name: stage11-waymo-one-item' || return $?
  _distil3d_accept_replace \
    "${DISTIL3D_ACCEPT_JOB_ROOT}/stage11-waymo-one-item/run.yaml" \
    selections/waymo/example-front.yaml \
    selections/local/stage11/waymo-one-item.yaml || return $?
  for name in r4c-lora-edm r4c-lora-lidar-depth; do
    _distil3d_accept_replace "${DISTIL3D_ACCEPT_JOB_ROOT}/${name}/run.yaml" \
      selections/waymo/example-r4c-split.yaml \
      selections/local/r4c-ddw85-split.yaml || return $?
  done
  printf 'forms\tbootstrap\tcopy tracked forms to ignored jobs/local/%s and bind fixed Stage 11 external roles\n' \
    "${DISTIL3D_ACCEPT_CAMPAIGN_KEY}" >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
}

_distil3d_accept_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

_distil3d_accept_gpu_foundation() {
  local output="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/nvidia-smi.txt"
  local map="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/gpu-map.tsv"
  local profile_id
  local row
  local index
  local uuid
  local model
  local memory
  local mig
  local matched
  local seen='|'
  local -a rows=()

  if [[ ${#DISTIL3D_PROFILE_GPU_IDS[@]} -ne 4 ]]; then
    printf 'a100-4 acceptance requires exactly four profile GPU IDs\n' >&2
    return 2
  fi
  if nvidia-smi \
    --query-gpu=index,uuid,name,memory.total,mig.mode.current \
    --format=csv,noheader,nounits > "${output}"
  then
    :
  else
    return $?
  fi
  while IFS= read -r row; do rows[${#rows[@]}]="${row}"; done < "${output}"
  printf 'profile_id\tindex\tuuid\tmodel\tmemory_mib\tmig\n' > "${map}"
  for profile_id in "${DISTIL3D_PROFILE_GPU_IDS[@]}"; do
    matched=false
    for row in "${rows[@]}"; do
      IFS=, read -r index uuid model memory mig <<< "${row}"
      index="$(_distil3d_accept_trim "${index}")"
      uuid="$(_distil3d_accept_trim "${uuid}")"
      model="$(_distil3d_accept_trim "${model}")"
      memory="$(_distil3d_accept_trim "${memory}")"
      mig="$(_distil3d_accept_trim "${mig}")"
      if [[ "${profile_id}" != "${index}" && "${profile_id}" != "${uuid}" ]]; then
        continue
      fi
      if [[ "${model}" != *A100* || "${mig}" != Disabled || \
        "${seen}" == *"|${uuid}|"* ]]
      then
        printf 'profile GPU is not one distinct A100 with MIG disabled: %s\n' \
          "${profile_id}" >&2
        return 2
      fi
      if [[ ! "${memory}" =~ ^[0-9]+$ ]] || (( 10#${memory} < 80000 )); then
        printf 'profile GPU is not one full A100 80 GB: %s\n' \
          "${profile_id}" >&2
        return 2
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${profile_id}" "${index}" "${uuid}" "${model}" "${memory}" "${mig}" \
        >> "${map}"
      seen+="${uuid}|"
      matched=true
      break
    done
    if [[ "${matched}" != true ]]; then
      printf 'profile GPU was not reported by nvidia-smi: %s\n' "${profile_id}" >&2
      return 2
    fi
  done
}

_distil3d_accept_require_selected_candidate() {
  case "${DISTIL3D_SELECTED_IMAGE_VARIANT}" in
    core)
      [[ "${DISTIL3D_SELECTED_IMAGE_ID}" == "${DISTIL3D_ACCEPT_CORE_IMAGE_ID}" ]]
      ;;
    moge)
      [[ "${DISTIL3D_SELECTED_IMAGE_ID}" == "${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" ]]
      ;;
    *)
      printf 'acceptance selected an image outside the candidate: %s\n' \
        "${DISTIL3D_SELECTED_IMAGE_VARIANT}" >&2
      return 2
      ;;
  esac
}

_distil3d_accept_image_foundation() {
  local variant
  local image_id
  local prefix

  for variant in core moge; do
    image_id="${DISTIL3D_ACCEPT_CORE_IMAGE_ID}"
    if [[ "${variant}" == moge ]]; then
      image_id="${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}"
    fi
    for prefix in core gen3c vggt; do
      docker run --rm --network none --entrypoint "" "${image_id}" \
        "/opt/envs/${prefix}/bin/python" -m novel_view.diagnostics.image \
        --prefix "${prefix}" --variant "${variant}" || return $?
    done
  done
}

_distil3d_accept_doctor_job() {
  local name="$1"
  local job_file

  job_file="$(distil3d_job_container_file "${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}/${name}")"
  distil3d_doctor_job "${job_file}" || return $?
  _distil3d_accept_require_selected_candidate
}

_distil3d_accept_foundation_body() {
  local raw="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/doctor.raw"
  local log="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/host/doctor.log"
  local doctor_status=0

  distil3d_load_profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" || return $?
  if distil3d_doctor_host > "${raw}" 2>&1; then
    :
  else
    doctor_status=$?
  fi
  while IFS= read -r line; do
    case "${line}" in
      docker.version=* | docker.architecture=* | docker.storage_driver=* | \
        docker.available_kib=* | profile.*_writable=ok)
        printf '%s\n' "${line}" >> "${log}"
        ;;
    esac
  done < "${raw}"
  rm -- "${raw}"
  if [[ ${doctor_status} -ne 0 ]]; then return "${doctor_status}"; fi
  if [[ ! -r "${DISTIL3D_DATA_ROOT}" || ! -r "${DISTIL3D_MODELS_ROOT}" ]]; then
    printf 'profile data/models roots are not readable\n' >&2
    return 2
  fi
  _distil3d_accept_gpu_foundation >> "${log}" 2>&1 || return $?
  _distil3d_accept_image_foundation >> "${log}" 2>&1 || return $?
  for name in euvs-source-views-v1 gaussian-full-sequence-v1 \
    gaussian-dense-overlap21-v1 stage11-waymo-one-item r4c-lora-edm \
    r4c-lora-lidar-depth
  do
    printf 'doctor-job=%s\n' "${name}" >> "${log}"
    _distil3d_accept_doctor_job "${name}" >> "${log}" 2>&1 || return $?
  done
}

_distil3d_accept_foundation() {
  local started
  local finished
  local status
  local bytes

  started="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'foundation\tfoundation\tdoctor host; exact four A100 without MIG; image and producer diagnostics\n' \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
  if _distil3d_accept_foundation_body; then
    status=0
  else
    status=$?
  fi
  finished="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  bytes=0
  for file in host/doctor.log host/nvidia-smi.txt host/gpu-map.tsv; do
    if [[ -f "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/${file}" ]]; then
      bytes=$((bytes + $(wc -c < "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/${file}")))
    fi
  done
  if [[ ${status} -eq 0 ]]; then
    _distil3d_accept_write_result foundation foundation true succeeded \
      succeeded 0 absent 0 false "${started}" "${finished}" none core+moge \
      "${DISTIL3D_ACCEPT_CORE_IMAGE_ID},${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" \
      "${bytes}" false 0 none 'host and shared diagnostics passed'
    return 0
  fi
  _distil3d_accept_write_result foundation foundation true failed absent \
    "${status}" absent null false "${started}" "${finished}" none core+moge \
    "${DISTIL3D_ACCEPT_CORE_IMAGE_ID},${DISTIL3D_ACCEPT_MOGE_IMAGE_ID}" \
    "${bytes}" false 0 none 'foundation failed'
  DISTIL3D_ACCEPT_REQUIRED_FAILED=1
  return "${status}"
}

_distil3d_accept_build_plan() {
  local name="$1"
  local job_file

  distil3d_load_profile "${DISTIL3D_ACCEPT_PROFILE_ARGUMENT}" || return $?
  job_file="$(distil3d_job_container_file "${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}/${name}")"
  distil3d_select_job_image "${job_file}" || return $?
  _distil3d_accept_require_selected_candidate || return $?
  distil3d_final_job_plan "${job_file}"
}

_distil3d_accept_plan_case() {
  local case_id="$1"
  local phase="$2"
  local required="$3"
  local name="$4"
  local key
  local case_root
  local status

  key="$(_distil3d_accept_case_key "${case_id}")"
  case_root="${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/cases/${key}"
  mkdir -p -- "${case_root}"
  printf '%s\tplan\t./distil3d plan --profile <profile> %s/%s\n' \
    "${case_id}" "${DISTIL3D_ACCEPT_JOB_RELATIVE_ROOT}" "${name}" \
    >> "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/commands.tsv"
  if _distil3d_accept_build_plan "${name}" \
    > "${DISTIL3D_ACCEPT_CAMPAIGN_ROOT}/plans/${key}.json" \
    2> "${case_root}/plan.log"
  then
    return 0
  else
    status=$?
  fi
  _distil3d_accept_write_no_attempt "${case_id}" "${phase}" "${required}" \
    failed "${status}" 'plan failed'
  return "${status}"
}
