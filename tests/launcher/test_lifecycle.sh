#!/usr/bin/env bash

set -euo pipefail

repository_root="$(
  CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P
)"
real_docker="$(type -P docker)"
task_root="$(mktemp -d "${TMPDIR:-/tmp}/distil3d-lifecycle.XXXXXX")"
task_token="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
jobs_root="${repository_root}/jobs/.lifecycle-${task_token}"
profile_root="${task_root}/profile"
run_root="${task_root}/runs"
container_ids=()

cleanup() {
  local container_id

  for container_id in "${container_ids[@]}"; do
    if [[ -n "${container_id}" ]]; then
      docker rm --force "${container_id}" >/dev/null 2>&1 || true
    fi
  done
  if [[ -d "${jobs_root}" ]]; then
    find "${jobs_root}" -depth -delete
  fi
  if [[ -d "${task_root}" ]]; then
    find "${task_root}" -depth -delete
  fi
}
trap cleanup EXIT

mkdir -p \
  "${task_root}/bin" \
  "${profile_root}" \
  "${task_root}/data" \
  "${task_root}/models" \
  "${task_root}/prepared" \
  "${run_root}" \
  "${task_root}/cache" \
  "${jobs_root}/success" \
  "${jobs_root}/failure" \
  "${jobs_root}/long" \
  "${jobs_root}/interrupt" \
  "${jobs_root}/stubborn" \
  "${jobs_root}/process-tree" \
  "${jobs_root}/ambiguous"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'arguments=("$@")' \
  'for index in "${!arguments[@]}"; do' \
  '  if [[ "${arguments[index]}" == "distil3d:cuda124-sm80-core" ]]; then' \
  '    arguments[index]="distil3d:cpu-test"' \
  '  fi' \
  'done' \
  'exec "${DISTIL3D_LIFECYCLE_REAL_DOCKER}" "${arguments[@]}"' \
  > "${task_root}/bin/docker"
chmod 755 "${task_root}/bin/docker"
export DISTIL3D_LIFECYCLE_REAL_DOCKER="${real_docker}"
export PATH="${task_root}/bin:${PATH}"

printf '%s\n' \
  "DISTIL3D_DATA_ROOT=${task_root}/data" \
  "DISTIL3D_MODELS_ROOT=${task_root}/models" \
  "DISTIL3D_PREPARED_ROOT=${task_root}/prepared" \
  "DISTIL3D_RUNS_ROOT=${run_root}" \
  "DISTIL3D_CACHE_ROOT=${task_root}/cache" \
  'DISTIL3D_GPU_IDS=' \
  > "${profile_root}/.env"

write_job() {
  local directory="$1"
  local job_name="$2"
  local mode="$3"
  local exit_code="$4"

  printf '%s\n' \
    'schema_version: 1' \
    "name: ${job_name}" \
    'workflow: {name: _lifecycle_smoke, version: 1}' \
    'input: {}' \
    'parameters:' \
    "  mode: ${mode}" \
    "  exit_code: ${exit_code}" \
    'execution: {preset: cpu_test}' \
    > "${directory}/run.yaml"
}

success_job="lifecycle-success-${task_token}"
failure_job="lifecycle-failure-${task_token}"
long_job="lifecycle-long-${task_token}"
interrupt_job="lifecycle-interrupt-${task_token}"
stubborn_job="lifecycle-stubborn-${task_token}"
process_tree_job="lifecycle-process-tree-${task_token}"
ambiguous_job="lifecycle-ambiguous-${task_token}"
write_job "${jobs_root}/success" "${success_job}" success 0
write_job "${jobs_root}/failure" "${failure_job}" failure 23
write_job "${jobs_root}/long" "${long_job}" long 0
write_job "${jobs_root}/interrupt" "${interrupt_job}" long 0
write_job "${jobs_root}/stubborn" "${stubborn_job}" stubborn 0
write_job "${jobs_root}/process-tree" "${process_tree_job}" process_tree 0
write_job "${jobs_root}/ambiguous" "${ambiguous_job}" long 0

container_for_ref() {
  docker ps --all --quiet \
    --filter "label=io.distil3d.attempt-ref=$1"
}

record_for_ref() {
  local reference="$1"
  local job_name="${reference%%/*}"
  local remainder="${reference#*/}"
  local run_id="${remainder%%/*}"
  local attempt_id="${reference##*/}"

  printf '%s/%s/%s/attempts/%s/attempt.json\n' \
    "${run_root}" "${job_name}" "${run_id}" "${attempt_id}"
}

wait_for_record() {
  local record_file

  record_file="$(record_for_ref "$1")"
  for _ in $(seq 1 200); do
    if [[ -f "${record_file}" ]]; then
      return 0
    fi
    sleep 0.05
  done
  printf 'attempt record did not appear: %s\n' "${record_file}" >&2
  return 1
}

wait_for_log() {
  local container_id="$1"
  local marker="$2"

  for _ in $(seq 1 200); do
    if docker logs "${container_id}" 2>&1 | grep -F "${marker}" >/dev/null; then
      return 0
    fi
    sleep 0.05
  done
  printf 'container log marker did not appear: %s\n' "${marker}" >&2
  return 1
}

container_host_pids() {
  docker top "$1" -eo pid,ppid,pgid,sid,comm | awk 'NR > 1 {print $1}'
}

assert_host_pids_gone() {
  local pid

  for pid in "$@"; do
    for _ in $(seq 1 100); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.05
    done
    if kill -0 "${pid}" 2>/dev/null; then
      printf 'container process is still alive: %s\n' "${pid}" >&2
      return 1
    fi
  done
}

assert_status() {
  local reference="$1"
  local recorded_state="$2"
  local container_state="$3"
  local container_exit_code="$4"
  local container_oom_killed="$5"
  local output_file="$6"

  timeout 5 "${repository_root}/distil3d" status \
    --profile "${profile_root}" \
    "${reference}" \
    > "${output_file}"
  grep -Fx "recorded_state=${recorded_state}" "${output_file}" >/dev/null
  grep -Fx "container_state=${container_state}" "${output_file}" >/dev/null
  grep -Fx "container_exit_code=${container_exit_code}" "${output_file}" >/dev/null
  grep -Fx "container_oom_killed=${container_oom_killed}" "${output_file}" >/dev/null
  grep -Fx "attempt_ref=${reference}" "${output_file}" >/dev/null
}

success_run="success-${task_token}"
success_output="${task_root}/success.log"
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/success" \
  --run-id "${success_run}" \
  > "${success_output}" 2>&1
success_container="$(
  docker ps --all --quiet \
    --filter "label=io.distil3d.job=${success_job}" \
    --filter "label=io.distil3d.run-id=${success_run}"
)"
container_ids+=("${success_container}")
test "$(docker inspect --format '{{.State.ExitCode}}' "${success_container}")" = 0
success_ref="$(
  docker inspect \
    --format '{{index .Config.Labels "io.distil3d.attempt-ref"}}' \
    "${success_container}"
)"

failure_run="failure-${task_token}"
failure_output="${task_root}/failure.log"
set +e
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/failure" \
  --run-id "${failure_run}" \
  > "${failure_output}" 2>&1
failure_status=$?
set -e
test "${failure_status}" -eq 23
failure_container="$(
  docker ps --all --quiet \
    --filter "label=io.distil3d.job=${failure_job}" \
    --filter "label=io.distil3d.run-id=${failure_run}"
)"
container_ids+=("${failure_container}")
test "$(docker inspect --format '{{.State.ExitCode}}' "${failure_container}")" = 23
failure_ref="$(
  docker inspect \
    --format '{{index .Config.Labels "io.distil3d.attempt-ref"}}' \
    "${failure_container}"
)"

long_run="long-${task_token}"
long_output="${task_root}/long.log"
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/long" \
  --run-id "${long_run}" \
  --detach \
  > "${long_output}" 2>&1
long_container="$(
  docker ps --all --quiet \
    --filter "label=io.distil3d.job=${long_job}" \
    --filter "label=io.distil3d.run-id=${long_run}"
)"
container_ids+=("${long_container}")
test "$(docker inspect --format '{{.State.Running}}' "${long_container}")" = true
long_ref="$(
  docker inspect \
    --format '{{index .Config.Labels "io.distil3d.attempt-ref"}}' \
    "${long_container}"
)"
wait_for_record "${long_ref}"

set +e
"${repository_root}/distil3d" remove \
  --profile "${profile_root}" \
  "${long_job}/${long_run}" \
  > "${task_root}/remove-running.out" \
  2> "${task_root}/remove-running.err"
remove_running_status=$?
set -e
test "${remove_running_status}" -eq 2
grep -F 'stop it first' "${task_root}/remove-running.err" >/dev/null
test "$(docker inspect --format '{{.State.Running}}' "${long_container}")" = true

interrupt_run="interrupt-${task_token}"
interrupt_output="${task_root}/interrupt.log"
set +e
timeout --preserve-status --signal=INT 3 \
  "${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/interrupt" \
  --run-id "${interrupt_run}" \
  > "${interrupt_output}" 2>&1
interrupt_status=$?
set -e
test "${interrupt_status}" -eq 130
interrupt_container="$(
  docker ps --all --quiet \
    --filter "label=io.distil3d.job=${interrupt_job}" \
    --filter "label=io.distil3d.run-id=${interrupt_run}"
)"
container_ids+=("${interrupt_container}")
test "$(docker inspect --format '{{.State.Running}}' "${interrupt_container}")" = true
grep -F 'recover: ./distil3d status' "${interrupt_output}" >/dev/null
grep -F 'recover: ./distil3d logs' "${interrupt_output}" >/dev/null

resume_output="${task_root}/resume.log"
"${repository_root}/distil3d" resume \
  --profile "${profile_root}" \
  "${success_ref}" \
  --checkpoint /models/operator-choice.pt \
  > "${resume_output}" 2>&1
mapfile -t success_containers < <(
  docker ps --all --quiet \
    --filter "label=io.distil3d.job=${success_job}" \
    --filter "label=io.distil3d.run-id=${success_run}"
)
test "${#success_containers[@]}" -eq 2
if [[ "${success_containers[0]}" == "${success_container}" ]]; then
  resume_container="${success_containers[1]}"
else
  resume_container="${success_containers[0]}"
fi
container_ids+=("${resume_container}")
test "$(docker inspect --format '{{.State.ExitCode}}' "${resume_container}")" = 0
resume_ref="$(
  docker inspect \
    --format '{{index .Config.Labels "io.distil3d.attempt-ref"}}' \
    "${resume_container}"
)"

resume_attempt_id="$(
  docker inspect \
    --format '{{index .Config.Labels "io.distil3d.attempt-id"}}' \
    "${resume_container}"
)"
resume_record="${run_root}/${success_job}/${success_run}/attempts/${resume_attempt_id}/attempt.json"
grep -F '"attempt_kind": "resume"' "${resume_record}" >/dev/null
grep -F '"selected_checkpoint": "/models/operator-choice.pt"' \
  "${resume_record}" >/dev/null

created_job="lifecycle-created-${task_token}"
created_run="created-${task_token}"
created_attempt="attempt-created-${task_token}"
created_ref="${created_job}/${created_run}/${created_attempt}"
created_container="$(
  docker create \
    --name "distil3d-created-${task_token}" \
    --label "io.distil3d.job=${created_job}" \
    --label "io.distil3d.run-id=${created_run}" \
    --label "io.distil3d.attempt-id=${created_attempt}" \
    --label "io.distil3d.attempt-ref=${created_ref}" \
    --entrypoint "" \
    distil3d:cpu-test \
    /bin/sh -c 'exit 0'
)"
container_ids+=("${created_container}")

absent_job="lifecycle-absent-${task_token}"
absent_run="absent-${task_token}"
absent_attempt="attempt-absent-${task_token}"
absent_ref="${absent_job}/${absent_run}/${absent_attempt}"
absent_container="$(
  docker create \
    --name "distil3d-absent-${task_token}" \
    --label "io.distil3d.job=${absent_job}" \
    --label "io.distil3d.run-id=${absent_run}" \
    --label "io.distil3d.attempt-id=${absent_attempt}" \
    --label "io.distil3d.attempt-ref=${absent_ref}" \
    --entrypoint "" \
    distil3d:cpu-test \
    /bin/sh -c 'exit 137'
)"
container_ids+=("${absent_container}")
docker start "${absent_container}" >/dev/null
test "$(docker wait "${absent_container}")" = 137

stale_run="stale-${task_token}"
stale_output="${task_root}/stale.log"
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/stubborn" \
  --run-id "${stale_run}" \
  --detach \
  > "${stale_output}" 2>&1
stale_ref="$(sed -n 's/^attempt=//p' "${stale_output}")"
stale_container="$(container_for_ref "${stale_ref}")"
container_ids+=("${stale_container}")
wait_for_record "${stale_ref}"
wait_for_log "${stale_container}" 'ready process_group='
mapfile -t stale_pids < <(container_host_pids "${stale_container}")
test "${#stale_pids[@]}" -eq 4
stale_record="$(record_for_ref "${stale_ref}")"
stale_record_hash="$(sha256sum "${stale_record}")"
"${repository_root}/distil3d" stop \
  --profile "${profile_root}" \
  "${stale_ref}" \
  --grace-seconds 1 \
  > "${task_root}/stop-stubborn.txt"
grep -Fx 'container_state=exited' "${task_root}/stop-stubborn.txt" >/dev/null
grep -Fx 'container_exit_code=137' "${task_root}/stop-stubborn.txt" >/dev/null
grep -Fx 'container_oom_killed=false' "${task_root}/stop-stubborn.txt" >/dev/null
test "$(sha256sum "${stale_record}")" = "${stale_record_hash}"
grep -F '"recorded_state": "running"' "${stale_record}" >/dev/null
assert_host_pids_gone "${stale_pids[@]}"

process_tree_run="process-tree-${task_token}"
process_tree_output="${task_root}/process-tree.log"
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/process-tree" \
  --run-id "${process_tree_run}" \
  --detach \
  > "${process_tree_output}" 2>&1
process_tree_ref="$(sed -n 's/^attempt=//p' "${process_tree_output}")"
process_tree_container="$(container_for_ref "${process_tree_ref}")"
container_ids+=("${process_tree_container}")
wait_for_record "${process_tree_ref}"
wait_for_log "${process_tree_container}" 'ready process_group='
mapfile -t process_tree_pids < <(container_host_pids "${process_tree_container}")
test "${#process_tree_pids[@]}" -eq 4
"${repository_root}/distil3d" stop \
  --profile "${profile_root}" \
  "${process_tree_ref}" \
  --grace-seconds 5 \
  > "${task_root}/stop-process-tree.txt"
grep -Fx 'container_state=exited' "${task_root}/stop-process-tree.txt" >/dev/null
grep -Fx 'container_exit_code=143' "${task_root}/stop-process-tree.txt" >/dev/null
grep -Fx 'container_oom_killed=false' "${task_root}/stop-process-tree.txt" >/dev/null
process_tree_record="$(record_for_ref "${process_tree_ref}")"
grep -F '"recorded_state": "stopped"' "${process_tree_record}" >/dev/null
grep -F '"worker_exit_code": 143' "${process_tree_record}" >/dev/null
docker logs "${process_tree_container}" > "${task_root}/process-tree-container.log" 2>&1
grep -F 'signal role=leader signal=SIGTERM' \
  "${task_root}/process-tree-container.log" >/dev/null
grep -F 'signal role=child signal=SIGTERM' \
  "${task_root}/process-tree-container.log" >/dev/null
grep -F 'signal role=grandchild signal=SIGTERM' \
  "${task_root}/process-tree-container.log" >/dev/null
assert_host_pids_gone "${process_tree_pids[@]}"

process_tree_attempt_root="$(dirname "${process_tree_record}")"
process_tree_record_hash="$(sha256sum "${process_tree_record}")"
"${repository_root}/distil3d" remove \
  --profile "${profile_root}" \
  "${process_tree_ref}" \
  > "${task_root}/remove-process-tree.txt"
test -z "$(container_for_ref "${process_tree_ref}")"
test -d "${process_tree_attempt_root}"
test -f "${process_tree_record}"
test "$(sha256sum "${process_tree_record}")" = "${process_tree_record_hash}"
"${repository_root}/distil3d" remove \
  --profile "${profile_root}" \
  "${process_tree_ref}" \
  > "${task_root}/remove-process-tree-again.txt"
grep -F 'container already absent' \
  "${task_root}/remove-process-tree-again.txt" >/dev/null

ambiguous_run="ambiguous-${task_token}"
ambiguous_a_output="${task_root}/ambiguous-a.log"
ambiguous_b_output="${task_root}/ambiguous-b.log"
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/ambiguous" \
  --run-id "${ambiguous_run}" \
  --detach \
  > "${ambiguous_a_output}" 2>&1
"${repository_root}/distil3d" run \
  --profile "${profile_root}" \
  "${jobs_root}/ambiguous" \
  --run-id "${ambiguous_run}" \
  --detach \
  > "${ambiguous_b_output}" 2>&1
ambiguous_a_ref="$(sed -n 's/^attempt=//p' "${ambiguous_a_output}")"
ambiguous_b_ref="$(sed -n 's/^attempt=//p' "${ambiguous_b_output}")"
ambiguous_a_container="$(container_for_ref "${ambiguous_a_ref}")"
ambiguous_b_container="$(container_for_ref "${ambiguous_b_ref}")"
container_ids+=("${ambiguous_a_container}" "${ambiguous_b_container}")
wait_for_record "${ambiguous_a_ref}"
wait_for_record "${ambiguous_b_ref}"

record_hashes_before="$(
  find "${run_root}" -name attempt.json -type f -exec sha256sum {} \; |
    LC_ALL=C sort
)"

assert_status \
  "${created_ref}" absent created null false \
  "${task_root}/status-created.txt"
assert_status \
  "${absent_ref}" absent exited 137 false \
  "${task_root}/status-absent-exited.txt"
assert_status \
  "${long_ref}" running running null false \
  "${task_root}/status-running.txt"
assert_status \
  "${success_ref}" succeeded exited 0 false \
  "${task_root}/status-succeeded.txt"
assert_status \
  "${failure_ref}" failed exited 23 false \
  "${task_root}/status-failed.txt"
assert_status \
  "${stale_ref}" running exited 137 false \
  "${task_root}/status-stale.txt"

"${repository_root}/distil3d" logs \
  --profile "${profile_root}" \
  "${success_ref}" \
  > "${task_root}/logs-success.txt"
grep -F 'lifecycle-smoke: success' "${task_root}/logs-success.txt" >/dev/null
"${repository_root}/distil3d" logs \
  --profile "${profile_root}" \
  "${success_ref}" \
  -f \
  > "${task_root}/logs-success-follow.txt"
grep -F 'lifecycle-smoke: success' \
  "${task_root}/logs-success-follow.txt" >/dev/null

docker rm "${success_container}" >/dev/null
assert_status \
  "${success_ref}" succeeded absent null null \
  "${task_root}/status-removed.txt"

timeout 5 "${repository_root}/distil3d" status \
  --profile "${profile_root}" \
  "${success_job}/${success_run}" \
  > "${task_root}/status-latest.txt"
grep -Fx "attempt_ref=${resume_ref}" "${task_root}/status-latest.txt" >/dev/null
grep -Fx 'recorded_state=succeeded' "${task_root}/status-latest.txt" >/dev/null

set +e
timeout 5 "${repository_root}/distil3d" status \
  --profile "${profile_root}" \
  "${ambiguous_job}/${ambiguous_run}" \
  > "${task_root}/status-ambiguous.out" \
  2> "${task_root}/status-ambiguous.err"
ambiguous_status=$?
set -e
test "${ambiguous_status}" -eq 2
grep -Fx "candidate=${ambiguous_a_ref}" \
  "${task_root}/status-ambiguous.err" >/dev/null
grep -Fx "candidate=${ambiguous_b_ref}" \
  "${task_root}/status-ambiguous.err" >/dev/null

record_hashes_after="$(
  find "${run_root}" -name attempt.json -type f -exec sha256sum {} \; |
    LC_ALL=C sort
)"
test "${record_hashes_before}" = "${record_hashes_after}"

stale_record_hash="$(sha256sum "${stale_record}")"
"${repository_root}/distil3d" remove \
  --profile "${profile_root}" \
  "${stale_ref}" \
  > "${task_root}/remove-stubborn.txt"
test -z "$(container_for_ref "${stale_ref}")"
test -f "${stale_record}"
test "$(sha256sum "${stale_record}")" = "${stale_record_hash}"

"${repository_root}/distil3d" stop \
  --profile "${profile_root}" \
  "${long_ref}" \
  --grace-seconds 5 \
  > "${task_root}/stop-long.txt"
"${repository_root}/distil3d" remove \
  --profile "${profile_root}" \
  "${long_ref}" \
  > "${task_root}/remove-long.txt"

printf '%s\n' \
  'success_exit=0' \
  'failure_exit=23' \
  'detach_running=true' \
  'ctrl_c_exit=130' \
  'ctrl_c_container_running=true' \
  'resume_exit=0' \
  'resume_checkpoint=/models/operator-choice.pt' \
  'status_matrix=passed' \
  'status_nonblocking=true' \
  'status_record_hashes_unchanged=true' \
  'logs_and_follow=passed' \
  'ambiguity_exit=2' \
  'graceful_stop=exited-143-oom-false' \
  'stubborn_stop=exited-137-oom-false' \
  'process_tree_cleanup=passed' \
  'remove_preserves_attempt=passed' \
  'running_remove_requires_stop=true'
