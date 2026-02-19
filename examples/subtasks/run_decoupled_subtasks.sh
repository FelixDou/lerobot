#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run PI05 eval then generate subtasks offline.

Usage:
  examples/subtasks/run_decoupled_subtasks.sh \
    --policy-path lerobot/pi05_libero_finetuned \
    --env-type libero \
    --n-episodes 5 \
    --batch-size 5 \
    --device cuda

Optional:
  --output-dir PATH              (default: outputs/eval/YYYY-MM-DD/HH-MM-SS_decoupled)
  --stride N                     (default: 1)
  --qwen-python PATH             (default: python)
  --backend BACKEND              (default: hf; choices: hf|openai)
  --model MODEL_ID               (default: Qwen/Qwen3-VL-4B-Instruct for hf, gpt-5.2 for openai)
  --dtype DTYPE                  (default: bfloat16)
  --openai-image-detail DETAIL   (default: auto; choices: auto|low|high)
  --openai-reasoning-effort EFFORT (optional: none|minimal|low|medium|high|xhigh)
  --temperature T                (default: 0.0)
  --max-new-tokens N             (default: 16)
  --image-key KEY                (default: empty, auto-pick)
EOF
}

POLICY_PATH=""
ENV_TYPE=""
N_EPISODES=""
BATCH_SIZE=""
DEVICE=""
OUTPUT_DIR=""
STRIDE="1"
QWEN_PYTHON="python"
BACKEND="hf"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
DTYPE="bfloat16"
OPENAI_IMAGE_DETAIL="auto"
OPENAI_REASONING_EFFORT=""
TEMPERATURE="0.0"
MAX_NEW_TOKENS="16"
IMAGE_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-path) POLICY_PATH="$2"; shift 2;;
    --env-type) ENV_TYPE="$2"; shift 2;;
    --n-episodes) N_EPISODES="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --stride) STRIDE="$2"; shift 2;;
    --qwen-python) QWEN_PYTHON="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --openai-image-detail) OPENAI_IMAGE_DETAIL="$2"; shift 2;;
    --openai-reasoning-effort) OPENAI_REASONING_EFFORT="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --image-key) IMAGE_KEY="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${POLICY_PATH}" || -z "${ENV_TYPE}" || -z "${N_EPISODES}" || -z "${BATCH_SIZE}" || -z "${DEVICE}" ]]; then
  echo "Missing required arguments."
  usage
  exit 1
fi

if [[ "${BACKEND}" != "hf" && "${BACKEND}" != "openai" ]]; then
  echo "Invalid --backend: ${BACKEND}. Must be hf or openai."
  exit 1
fi

if [[ "${BACKEND}" == "openai" && "${MODEL}" == "Qwen/Qwen3-VL-4B-Instruct" ]]; then
  MODEL="gpt-5.2"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  ts_dir="$(date +%Y-%m-%d)/$(date +%H-%M-%S)_decoupled"
  OUTPUT_DIR="outputs/eval/${ts_dir}"
fi

echo "Running eval to ${OUTPUT_DIR}..."
lerobot-eval \
  --policy.path="${POLICY_PATH}" \
  --env.type="${ENV_TYPE}" \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.batch_size="${BATCH_SIZE}" \
  --eval.export_subtask_inputs=true \
  --eval.subtask_inputs_stride="${STRIDE}" \
  --policy.device="${DEVICE}" \
  --output_dir="${OUTPUT_DIR}"

INPUTS_DIR="${OUTPUT_DIR}/subtask_inputs"
OUTPUT_SUBTASKS_DIR="${OUTPUT_DIR}/subtasks_${BACKEND}"

echo "Generating subtasks into ${OUTPUT_SUBTASKS_DIR}..."
CMD=(
  "${QWEN_PYTHON}"
  "examples/subtasks/generate_subtasks_offline.py"
  --inputs-dir "${INPUTS_DIR}"
  --output-dir "${OUTPUT_SUBTASKS_DIR}"
  --backend "${BACKEND}"
  --model "${MODEL}"
  --dtype "${DTYPE}"
  --temperature "${TEMPERATURE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
if [[ "${BACKEND}" == "openai" ]]; then
  CMD+=(--openai-image-detail "${OPENAI_IMAGE_DETAIL}")
  if [[ -n "${OPENAI_REASONING_EFFORT}" ]]; then
    CMD+=(--openai-reasoning-effort "${OPENAI_REASONING_EFFORT}")
  fi
fi
if [[ -n "${IMAGE_KEY}" ]]; then
  CMD+=(--image-key "${IMAGE_KEY}")
fi

"${CMD[@]}"

echo "Done."
