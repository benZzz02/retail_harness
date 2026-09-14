#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="$project_dir/.external/tau-bench"
venv_dir="$project_dir/.venv"
repository="https://github.com/sierra-research/tau-bench.git"
commit="59a200c6d575d595120f1cb70fea53cef0632f6b"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif command -v python3.11 >/dev/null 2>&1; then
  python_bin="python3.11"
else
  python_bin="python3"
fi

"$python_bin" -c 'import sys; assert sys.version_info >= (3, 10), "tau-bench requires Python 3.10+"'

created_checkout=0
if [[ ! -d "$external_dir/.git" ]]; then
  mkdir -p "$(dirname "$external_dir")"
  git clone "$repository" "$external_dir"
  created_checkout=1
fi

actual_repository="$(git -C "$external_dir" remote get-url origin)"
if [[ "$actual_repository" != "$repository" ]]; then
  echo "Unexpected tau-bench origin: $actual_repository" >&2
  exit 1
fi

if [[ "$created_checkout" == "1" ]]; then
  git -C "$external_dir" checkout --detach "$commit"
fi

actual_commit="$(git -C "$external_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$commit" ]]; then
  echo "Existing external checkout is not at the pinned commit." >&2
  echo "Expected: $commit" >&2
  echo "Actual:   $actual_commit" >&2
  echo "Move or explicitly update .external/tau-bench, then rerun setup." >&2
  exit 1
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 "$venv_dir/bin/python" -m pip install \
  -e "$external_dir" -e "$project_dir"
LITELLM_LOCAL_MODEL_COST_MAP=True "$venv_dir/bin/python" -c \
  'from tau_bench.envs.retail.env import MockRetailDomainEnv; env = MockRetailDomainEnv(user_strategy="human", user_model="unused", task_split="test", task_index=0); assert len(env.data["orders"]) == 1000 and len(env.tools_info) == 16'
echo "tau-bench Retail is ready: $venv_dir/bin/python"
