#!/usr/bin/env bash
set -euo pipefail

# Colab bootstrap for running tau2-bench airline with:
# - local Qwen3-8B as the agent via Ollama
# - OpenAI GPT as the simulated user / evaluator
#
# Required:
#   export OPENAI_API_KEY="sk-..."
#
# Optional for private repos:
#   export GITHUB_TOKEN="github_pat_..."
#
# Optional knobs:
#   export REPO_URL="https://github.com/KoSpades/reinforcement_learning.git"
#   export REPO_DIR="/content/reinforcement_learning"
#   export AGENT_MODEL="ollama_chat/qwen3:8b"
#   export OLLAMA_MODEL="qwen3:8b"
#   export USER_MODEL="gpt-4.1-2025-04-14"
# Usage:
#   bash colab_tau2_airline_qwen.sh
#   bash colab_tau2_airline_qwen.sh --num-tasks 5 --runs 2
#   bash colab_tau2_airline_qwen.sh --save-prefix experiment-a
#
# Defaults:
#   --num-tasks all airline tasks
#   --runs 1
#   --save-prefix colab-qwen-agent-airline-<timestamp>

REPO_URL="${REPO_URL:-https://github.com/KoSpades/reinforcement_learning.git}"
REPO_DIR="${REPO_DIR:-/content/reinforcement_learning}"
POST_TRAIN_DIR="${REPO_DIR}/rl_post_train"
TAU2_DIR="${POST_TRAIN_DIR}/external/tau2-bench"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:8b}"
AGENT_MODEL="${AGENT_MODEL:-ollama_chat/${OLLAMA_MODEL}}"
USER_MODEL="${USER_MODEL:-gpt-4.1-2025-04-14}"
NUM_TASKS="${NUM_TASKS:-}"
RUNS="${RUNS:-1}"
SAVE_PREFIX="${SAVE_PREFIX:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
Usage: bash colab_tau2_airline_qwen.sh [options]

Options:
  --num-tasks N        Number of airline tasks to run per run. Omit for all tasks.
  --runs N            Number of complete runs to execute. Default: 1.
  --save-prefix NAME  Prefix for output directories. Default: colab-qwen-agent-airline-<timestamp>.
  --help              Show this help.

Environment:
  OPENAI_API_KEY      Required.
  GITHUB_TOKEN        Optional, for private repo clone.
  OLLAMA_MODEL        Default: qwen3:8b.
  AGENT_MODEL         Default: ollama_chat/${OLLAMA_MODEL}.
  USER_MODEL          Default: gpt-4.1-2025-04-14.
  MAX_CONCURRENCY     Default: 1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-tasks)
      NUM_TASKS="${2:?Missing value for --num-tasks}"
      shift 2
      ;;
    --runs)
      RUNS="${2:?Missing value for --runs}"
      shift 2
      ;;
    --save-prefix)
      SAVE_PREFIX="${2:?Missing value for --save-prefix}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
  echo "ERROR: --runs must be a positive integer." >&2
  exit 1
fi

if [[ -n "${NUM_TASKS}" ]] && ! [[ "${NUM_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --num-tasks must be a positive integer, or omit it to run all tasks." >&2
  exit 1
fi

if [[ -z "${SAVE_PREFIX}" ]]; then
  SAVE_PREFIX="colab-qwen-agent-airline-${RUN_TIMESTAMP}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "Run: export OPENAI_API_KEY='sk-...'" >&2
  exit 1
fi

echo "==> Installing system dependencies"
apt-get update
apt-get install -y curl git zstd

echo "==> Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> Installing Ollama if needed"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> Cloning or updating repo"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  CLONE_URL="${REPO_URL}"
  if [[ -n "${GITHUB_TOKEN:-}" && "${REPO_URL}" == https://github.com/* ]]; then
    CLONE_URL="${REPO_URL/https:\/\/github.com\//https:\/\/${GITHUB_TOKEN}@github.com\/}"
  fi
  git clone --recurse-submodules "${CLONE_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" pull --ff-only || true
  git -C "${REPO_DIR}" submodule update --init --recursive || true
fi

echo "==> Checking tau2 directory"
if [[ ! -d "${TAU2_DIR}" ]]; then
  echo "ERROR: tau2-bench directory not found at ${TAU2_DIR}" >&2
  exit 1
fi

echo "==> Starting Ollama"
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 8
fi
curl -fsS http://localhost:11434/api/tags >/dev/null

echo "==> Pulling ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}"

echo "==> Installing tau2 dependencies"
cd "${TAU2_DIR}"
uv sync

echo "==> Writing tau2 .env"
cp -n .env.example .env
python - <<'PY'
import os
from pathlib import Path

env_path = Path(".env")
lines = env_path.read_text().splitlines() if env_path.exists() else []
lines = [line for line in lines if not line.startswith("OPENAI_API_KEY=")]
lines.append(f"OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}")
env_path.write_text("\n".join(lines) + "\n")
PY

echo "==> Verifying tau2 data"
uv run tau2 check-data

echo "==> Verifying LiteLLM -> Ollama chat path"
uv run python - <<PY
from litellm import completion

response = completion(
    model="${AGENT_MODEL}",
    api_base="http://localhost:11434",
    messages=[{"role": "user", "content": "Reply with exactly OK. /no_think"}],
    temperature=0,
    timeout=120,
)
print(response.choices[0].message.content)
PY

echo "==> Running tau2 airline"
for run_idx in $(seq 1 "${RUNS}"); do
  SAVE_TO="${SAVE_PREFIX}-run-${run_idx}"
  echo "==> Run ${run_idx}/${RUNS}; save_to=${SAVE_TO}"

  RUN_ARGS=(
    uv run tau2 run
    --domain airline
    --agent-llm "${AGENT_MODEL}"
    --agent-llm-args '{"api_base":"http://localhost:11434","temperature":0.0}'
    --user-llm "${USER_MODEL}"
    --num-trials 1
    --max-concurrency "${MAX_CONCURRENCY}"
    --save-to "${SAVE_TO}"
  )

  if [[ -n "${NUM_TASKS}" ]]; then
    RUN_ARGS+=(--num-tasks "${NUM_TASKS}")
  fi

  "${RUN_ARGS[@]}"
  echo "Results: ${TAU2_DIR}/data/simulations/${SAVE_TO}"
done

echo "==> Done"
echo "All results are under: ${TAU2_DIR}/data/simulations/${SAVE_PREFIX}-run-*"
