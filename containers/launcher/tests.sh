#!/usr/bin/env bash

distil3d_test_usage() {
  cat <<'EOF'
Usage:
  ./distil3d test [unit|contract|integration|all]
EOF
}

distil3d_test_main() {
  local test_group="all"
  local -a test_paths

  if [[ $# -gt 1 ]]; then
    distil3d_test_usage >&2
    return 2
  fi
  if [[ $# -eq 1 ]]; then
    case "$1" in
      unit | contract | integration | all)
        test_group="$1"
        ;;
      -h | --help | help)
        distil3d_test_usage
        return 0
        ;;
      *)
        printf 'unknown test group: %s\n' "$1" >&2
        distil3d_test_usage >&2
        return 2
        ;;
    esac
  fi

  distil3d_build_cpu_test

  case "${test_group}" in
    unit)
      test_paths=(tests/unit)
      ;;
    contract)
      test_paths=(tests/contract)
      ;;
    integration)
      test_paths=(
        tests/integration
        /workspace/tests/launcher/test_distil3d.py
      )
      ;;
    all)
      test_paths=(
        tests/unit
        tests/contract
        tests/integration
        /workspace/tests/launcher/test_distil3d.py
      )
      ;;
  esac

  distil3d_run_cpu_test \
    /opt/envs/core/bin/python \
    -m pytest \
    -q \
    -m "not native_api" \
    "${test_paths[@]}"

  if [[ "${test_group}" == "all" ]]; then
    "${DISTIL3D_REPOSITORY_ROOT}/tests/launcher/test_lifecycle.sh"
  fi
}
