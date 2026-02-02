#!/bin/bash
# Updated training command for pi0 fine-tuning with:
# - No in-training evals (eval_freq=-1) or a single episode if enabled
# - Third phase: resume from 30k checkpoint for +40k more, ckpts every 2k

OUT=./outputs/pi0_libero_ft_my_run_20251215-215925  # existing run to resume
LOG=./train_pi0_third_$(date +%Y%m%d-%H%M%S).out
# Path to the saved train_config.json from the 30k checkpoint
CONFIG_PATH="$OUT/checkpoints/030000/pretrained_model/train_config.json"

# Get HF_USER from environment or use default
HF_USER=${HF_USER:-"your_hf_username"}
MASTER_PORT=${MASTER_PORT:-29515}

# Set your desired GPUs here (comma-separated), e.g., "0,1" or "2,3"
GPU_LIST="4,5,6,7"
# Number of processes must match the number of GPUs listed above
NPROC=4
# Avoid locking GPUs for others; we unset first, then set to your list
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASTER_PORT=$MASTER_PORT \
MUJOCO_GL=egl \
# Log the key env vars for debugging (separate command to avoid parsing torchrun args)
env | grep -E 'TIMEOUT|NCCL|MASTER_PORT'
nohup torchrun --master_port=$MASTER_PORT --nproc_per_node=$NPROC src/lerobot/scripts/lerobot_train.py \
  --policy.type=pi0 \
  --policy.pretrained_path=lerobot/pi0_libero \
  --policy.repo_id=${HF_USER}/pi0_libero_ft_my_run_custom \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=14 \
  --resume=true \
  --config_path="$CONFIG_PATH" \
  --steps=70000 \
  --save_freq=2000 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=-1 \
  --output_dir="$OUT" \
  > "$LOG" 2>&1 &

echo "Training started with PID: $!"
echo "Output directory: $OUT"
echo "Log file: $LOG"
echo "Monitor with: tail -f $LOG"

