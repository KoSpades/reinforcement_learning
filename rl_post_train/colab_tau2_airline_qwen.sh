#!/usr/bin/env bash
set -euo pipefail

# Colab bootstrap for running tau2-bench airline with:
# - Qwen3-8B as the agent via vLLM on Colab/cloud
# - Qwen3-8B as the agent via Ollama on local machines
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
#   export AGENT_BACKEND="vllm"  # or "ollama"
#   export VLLM_MODEL="Qwen/Qwen3-8B"
#   export OLLAMA_MODEL="qwen3:8b"
#   export USER_MODEL="gpt-4.1-2025-04-14"
#   export AGENT_TIMEOUT="60"
#   export AGENT_NUM_RETRIES="0"
# Usage:
#   bash colab_tau2_airline_qwen.sh
#   bash colab_tau2_airline_qwen.sh --num-tasks 5 --runs 2
#   bash colab_tau2_airline_qwen.sh --save-prefix experiment-a
#
# Defaults:
#   --num-tasks all airline tasks
#   --runs 1
#   --save-prefix colab-qwen-agent-airline-<timestamp>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d /content ]]; then
  DEFAULT_REPO_DIR="/content/reinforcement_learning"
else
  DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

REPO_URL="${REPO_URL:-https://github.com/KoSpades/reinforcement_learning.git}"
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"
POST_TRAIN_DIR="${REPO_DIR}/rl_post_train"
TAU2_DIR="${POST_TRAIN_DIR}/external/tau2-bench"

if [[ -z "${AGENT_BACKEND:-}" ]]; then
  if [[ -n "${COLAB_RELEASE_TAG:-}" || -d /content ]]; then
    AGENT_BACKEND="vllm"
  else
    AGENT_BACKEND="ollama"
  fi
fi

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
VLLM_SERVED_MODEL="${VLLM_SERVED_MODEL:-qwen3-8b}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:${VLLM_PORT}/v1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:8b}"
USER_MODEL="${USER_MODEL:-gpt-4.1-2025-04-14}"
NUM_TASKS="${NUM_TASKS:-}"
RUNS="${RUNS:-1}"
SAVE_PREFIX="${SAVE_PREFIX:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-60}"
AGENT_NUM_RETRIES="${AGENT_NUM_RETRIES:-0}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIRS=()

case "${AGENT_BACKEND}" in
  vllm)
    AGENT_MODEL="${AGENT_MODEL:-openai/${VLLM_SERVED_MODEL}}"
    AGENT_API_BASE="${AGENT_API_BASE:-${VLLM_API_BASE}}"
    AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"
    ;;
  ollama)
    AGENT_MODEL="${AGENT_MODEL:-ollama_chat/${OLLAMA_MODEL}}"
    AGENT_API_BASE="${AGENT_API_BASE:-http://localhost:11434}"
    AGENT_API_KEY="${AGENT_API_KEY:-}"
    ;;
  *)
    echo "ERROR: AGENT_BACKEND must be either 'vllm' or 'ollama'." >&2
    exit 1
    ;;
esac

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
  AGENT_BACKEND       Default: vllm on Colab/cloud, ollama locally.
  VLLM_MODEL          Default: Qwen/Qwen3-8B.
  VLLM_SERVED_MODEL   Default: qwen3-8b.
  VLLM_PORT           Default: 8000.
  VLLM_TOOL_CALL_PARSER Default: hermes.
  OLLAMA_MODEL        Default: qwen3:8b.
  AGENT_MODEL         Default: openai/${VLLM_SERVED_MODEL} for vllm, ollama_chat/${OLLAMA_MODEL} for ollama.
  USER_MODEL          Default: gpt-4.1-2025-04-14.
  MAX_CONCURRENCY     Default: 1.
  AGENT_TIMEOUT       Timeout in seconds for local agent calls. Default: 60.
  AGENT_NUM_RETRIES   LiteLLM retry count for local agent calls. Default: 0.
  PUSH_RESULTS        Commit and push generated results. Default: 0 unless GITHUB_TOKEN is set.
EOF
}

agent_llm_args_json() {
  if [[ -n "${AGENT_API_KEY}" ]]; then
    printf '{"api_base":"%s","api_key":"%s","temperature":0.0,"timeout":%s,"num_retries":%s}' \
      "${AGENT_API_BASE}" \
      "${AGENT_API_KEY}" \
      "${AGENT_TIMEOUT}" \
      "${AGENT_NUM_RETRIES}"
  else
    printf '{"api_base":"%s","temperature":0.0,"timeout":%s,"num_retries":%s}' \
      "${AGENT_API_BASE}" \
      "${AGENT_TIMEOUT}" \
      "${AGENT_NUM_RETRIES}"
  fi
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
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y curl git zstd
else
  echo "apt-get not found; assuming curl, git, and zstd are already available."
fi

echo "==> Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
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

echo "==> Installing tau2 dependencies"
cd "${TAU2_DIR}"
uv sync

if [[ "${AGENT_BACKEND}" == "ollama" ]]; then
  echo "==> Installing Ollama if needed"
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  echo "==> Starting Ollama"
  if ! curl -fsS "${AGENT_API_BASE}/api/tags" >/dev/null 2>&1; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 8
  fi
  curl -fsS "${AGENT_API_BASE}/api/tags" >/dev/null

  echo "==> Pulling ${OLLAMA_MODEL}"
  ollama pull "${OLLAMA_MODEL}"
else
  echo "==> Installing vLLM"
  uv pip install vllm

  echo "==> Starting vLLM"
  VLLM_HEALTH_URL="${VLLM_API_BASE%/v1}/health"
  if ! curl -fsS "${VLLM_HEALTH_URL}" >/dev/null 2>&1; then
    VLLM_CMD=(
      uv run vllm serve "${VLLM_MODEL}"
      --served-model-name "${VLLM_SERVED_MODEL}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --enable-auto-tool-choice \
      --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
    )
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
      read -r -a VLLM_EXTRA_ARGS_ARRAY <<< "${VLLM_EXTRA_ARGS}"
      VLLM_CMD+=("${VLLM_EXTRA_ARGS_ARRAY[@]}")
    fi
    nohup "${VLLM_CMD[@]}" > /tmp/vllm.log 2>&1 &
  fi

  for attempt in $(seq 1 120); do
    if curl -fsS "${VLLM_HEALTH_URL}" >/dev/null 2>&1; then
      break
    fi
    if [[ "${attempt}" == "120" ]]; then
      echo "ERROR: vLLM did not become healthy. Last log lines:" >&2
      tail -n 80 /tmp/vllm.log >&2 || true
      exit 1
    fi
    sleep 5
  done
fi

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

echo "==> Verifying LiteLLM -> ${AGENT_BACKEND} chat path"
uv run python - <<PY
from litellm import completion

response = completion(
    model="${AGENT_MODEL}",
    api_base="${AGENT_API_BASE}",
    **({"api_key": "${AGENT_API_KEY}"} if "${AGENT_API_KEY}" else {}),
    messages=[{"role": "user", "content": "Reply with exactly OK. /no_think"}],
    temperature=0,
    timeout=120,
)
print(response.choices[0].message.content)
PY

echo "==> Verifying LiteLLM -> ${AGENT_BACKEND} tool path"
uv run python - <<PY
from litellm import completion

response = completion(
    model="${AGENT_MODEL}",
    api_base="${AGENT_API_BASE}",
    **({"api_key": "${AGENT_API_KEY}"} if "${AGENT_API_KEY}" else {}),
    messages=[
        {
            "role": "user",
            "content": "Use the lookup_reservation tool for reservation ABC123. /no_think",
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "lookup_reservation",
                "description": "Look up an airline reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {"type": "string"}
                    },
                    "required": ["reservation_id"],
                },
            },
        }
    ],
    tool_choice="auto",
    temperature=0,
    timeout=120,
)

message = response.choices[0].message
if not getattr(message, "tool_calls", None):
    raise SystemExit(f"Expected a tool call, got: {message}")
print(message.tool_calls[0].function.name)
PY

echo "==> Running tau2 airline"
for run_idx in $(seq 1 "${RUNS}"); do
  SAVE_TO="${SAVE_PREFIX}-run-${run_idx}"
  echo "==> Run ${run_idx}/${RUNS}; save_to=${SAVE_TO}"

  RUN_ARGS=(
    uv run tau2 run
    --domain airline
    --agent-llm "${AGENT_MODEL}"
    --agent-llm-args "$(agent_llm_args_json)"
    --user-llm "${USER_MODEL}"
    --num-trials 1
    --max-retries 0
    --max-concurrency "${MAX_CONCURRENCY}"
    --save-to "${SAVE_TO}"
  )

  if [[ -n "${NUM_TASKS}" ]]; then
    RUN_ARGS+=(--num-tasks "${NUM_TASKS}")
  fi

  "${RUN_ARGS[@]}"
  echo "Results: ${TAU2_DIR}/data/simulations/${SAVE_TO}"
  RESULT_DIRS+=("${TAU2_DIR}/data/simulations/${SAVE_TO}")
done

echo "==> Done"
echo "All results are under: ${TAU2_DIR}/data/simulations/${SAVE_PREFIX}-run-*"

if [[ "${PUSH_RESULTS}" == "1" || -n "${GITHUB_TOKEN:-}" ]]; then
  echo "==> Committing and pushing results"
  cd "${REPO_DIR}"

  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "No GITHUB_TOKEN set; skipping git push."
    exit 0
  fi

  if [[ "${REPO_URL}" == https://github.com/* ]]; then
    PUSH_URL="${REPO_URL/https:\/\/github.com\//https:\/\/${GITHUB_TOKEN}@github.com\/}"
    git remote set-url origin "${PUSH_URL}"
  fi

  git config user.name "${GIT_AUTHOR_NAME:-Colab Tau2 Runner}"
  git config user.email "${GIT_AUTHOR_EMAIL:-colab-tau2-runner@example.com}"

  for result_dir in "${RESULT_DIRS[@]}"; do
    git add -f "${result_dir}"
  done

  if git diff --cached --quiet; then
    echo "No result changes to commit."
  else
    git commit -m "Add tau2 airline results ${SAVE_PREFIX}"
    git push origin HEAD
  fi
else
  echo "==> Skipping git push because no GITHUB_TOKEN was provided"
fi
