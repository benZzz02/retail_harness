#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${1:-$project_dir/.refundpilot_runs/dialogue-test-0-9}"
log_dir="$runtime_dir/stdout"
passed=0
failed=0

mkdir -p "$log_dir"

for task_index in 0 1 2 3 4 5 6 7 8 9; do
  echo
  echo "=== DIALOGUE TASK $task_index/9 ==="
  PYTHONUNBUFFERED=1 "$project_dir/.venv/bin/python" -m refundpilot \
    --runtime-dir "$runtime_dir" \
    tau-dialogue \
    --split test \
    --task "$task_index" \
    --agent-model gpt-5.6-luna \
    --user-model gpt-5.6-luna \
    --reasoning low \
    --max-steps 30 2>&1 | tee "$log_dir/task-$task_index.log"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
  echo "DIALOGUE PROGRESS: passed=$passed failed=$failed completed=$((task_index + 1))/10"
done

echo
echo "=== DIALOGUE COMPLETE: passed=$passed failed=$failed total=10 ==="
if [[ "$failed" -eq 0 ]]; then
  exit 0
fi
exit 1
