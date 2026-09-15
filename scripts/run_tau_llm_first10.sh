#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_dir/.refundpilot_runs/llm-test-0-9"
passed=0
failed=0

for task_index in 0 1 2 3 4 5 6 7 8 9; do
  echo
  echo "=== BATCH TASK $task_index/9 ==="
  if PYTHONUNBUFFERED=1 "$project_dir/.venv/bin/python" -m refundpilot \
    --runtime-dir "$runtime_dir" \
    tau-llm \
    --split test \
    --task "$task_index" \
    --model gpt-5.6-luna \
    --reasoning low \
    --max-steps 20 \
    --context-mode compact \
    --gate-mode guarded \
    --max-model-calls 16; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
  fi
  echo "BATCH PROGRESS: passed=$passed failed=$failed completed=$((task_index + 1))/10"
done

echo
echo "=== BATCH COMPLETE: passed=$passed failed=$failed total=10 ==="
if [[ "$failed" -eq 0 ]]; then
  exit 0
fi
exit 1
