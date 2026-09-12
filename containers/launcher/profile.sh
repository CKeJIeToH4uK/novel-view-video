#!/usr/bin/env bash

# Профиль читается как данные. Значения после первого `=` остаются буквальными.
distil3d_load_profile() {
  local profile_argument="$1"
  local profile_directory
  local profile_file
  local profile_line
  local profile_key
  local profile_value
  local seen_keys="|"
  local gpu_id
  local previous_gpu_id

  if [[ "${profile_argument}" == /* ]]; then
    profile_directory="${profile_argument}"
  else
    profile_directory="${DISTIL3D_REPOSITORY_ROOT}/${profile_argument}"
  fi
  profile_file="${profile_directory}/.env"

  DISTIL3D_PROFILE_DIRECTORY="${profile_directory}"
  DISTIL3D_DATA_ROOT=""
  DISTIL3D_MODELS_ROOT=""
  DISTIL3D_PREPARED_ROOT=""
  DISTIL3D_RUNS_ROOT=""
  DISTIL3D_CACHE_ROOT=""
  DISTIL3D_GPU_IDS=""

  while IFS= read -r profile_line || [[ -n "${profile_line}" ]]; do
    if [[ -z "${profile_line}" || "${profile_line}" == \#* ]]; then
      continue
    fi
    if [[ "${profile_line}" != *=* ]]; then
      printf 'profile line must be KEY=VALUE: %s\n' "${profile_line}" >&2
      return 2
    fi

    profile_key="${profile_line%%=*}"
    profile_value="${profile_line#*=}"
    if [[ "${seen_keys}" == *"|${profile_key}|"* ]]; then
      printf 'duplicate profile key: %s\n' "${profile_key}" >&2
      return 2
    fi

    case "${profile_key}" in
      DISTIL3D_DATA_ROOT)
        DISTIL3D_DATA_ROOT="${profile_value}"
        ;;
      DISTIL3D_MODELS_ROOT)
        DISTIL3D_MODELS_ROOT="${profile_value}"
        ;;
      DISTIL3D_PREPARED_ROOT)
        DISTIL3D_PREPARED_ROOT="${profile_value}"
        ;;
      DISTIL3D_RUNS_ROOT)
        DISTIL3D_RUNS_ROOT="${profile_value}"
        ;;
      DISTIL3D_CACHE_ROOT)
        DISTIL3D_CACHE_ROOT="${profile_value}"
        ;;
      DISTIL3D_GPU_IDS)
        DISTIL3D_GPU_IDS="${profile_value}"
        ;;
      *)
        printf 'unknown profile key: %s\n' "${profile_key}" >&2
        return 2
        ;;
    esac
    seen_keys+="${profile_key}|"
  done < "${profile_file}"

  for profile_key in DISTIL3D_DATA_ROOT DISTIL3D_MODELS_ROOT \
    DISTIL3D_PREPARED_ROOT DISTIL3D_RUNS_ROOT DISTIL3D_CACHE_ROOT DISTIL3D_GPU_IDS
  do
    if [[ "${seen_keys}" != *"|${profile_key}|"* ]]; then
      printf 'missing profile key: %s\n' "${profile_key}" >&2
      return 2
    fi
  done

  DISTIL3D_PROFILE_GPU_IDS=()
  if [[ -n "${DISTIL3D_GPU_IDS}" ]]; then
    case "${DISTIL3D_GPU_IDS}" in
      ,* | *, | *,,*)
        printf 'GPU list must not contain empty elements\n' >&2
        return 2
        ;;
    esac
    local -a profile_gpu_ids
    IFS=, read -r -a profile_gpu_ids <<< "${DISTIL3D_GPU_IDS}"
    for gpu_id in "${profile_gpu_ids[@]}"; do
      for previous_gpu_id in "${DISTIL3D_PROFILE_GPU_IDS[@]}"; do
        if [[ "${gpu_id}" == "${previous_gpu_id}" ]]; then
          printf 'duplicate GPU ID: %s\n' "${gpu_id}" >&2
          return 2
        fi
      done
      DISTIL3D_PROFILE_GPU_IDS+=("${gpu_id}")
    done
  fi
}

# Выбор первых N устройств из профиля, без обращения к GPU или Docker.
distil3d_select_gpu_resources() {
  local gpu_count="$1"
  local index
  local separator=""

  DISTIL3D_SELECTED_HOST_GPU_IDS=""
  DISTIL3D_SELECTED_CONTAINER_GPU_INDICES=""
  DISTIL3D_SELECTED_IPC=""
  DISTIL3D_SELECTED_MEMLOCK=""
  DISTIL3D_RESOURCE_ARGUMENTS=(--network none)
  if (( gpu_count > ${#DISTIL3D_PROFILE_GPU_IDS[@]} )); then
    printf 'profile has fewer GPU IDs than the execution preset requires\n' >&2
    return 2
  fi
  for (( index=0; index<gpu_count; index++ )); do
    DISTIL3D_SELECTED_HOST_GPU_IDS+="${separator}${DISTIL3D_PROFILE_GPU_IDS[index]}"
    DISTIL3D_SELECTED_CONTAINER_GPU_INDICES+="${separator}${index}"
    separator=,
  done
  if (( gpu_count > 0 )); then
    DISTIL3D_SELECTED_IPC=host
    DISTIL3D_SELECTED_MEMLOCK=-1:-1
    DISTIL3D_RESOURCE_ARGUMENTS+=(
      --gpus "\"device=${DISTIL3D_SELECTED_HOST_GPU_IDS}\""
      --env "CUDA_VISIBLE_DEVICES=${DISTIL3D_SELECTED_CONTAINER_GPU_INDICES}"
      --ipc "${DISTIL3D_SELECTED_IPC}"
      --ulimit "memlock=${DISTIL3D_SELECTED_MEMLOCK}"
    )
  fi
}
