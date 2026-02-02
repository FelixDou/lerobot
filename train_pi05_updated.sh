#!/bin/bash
# Updated training command for pi05 fine-tuning with:
# - No in-training evals (eval_freq=-1) or a single episode if enabled
# - save_freq=1000 (checkpoints every 1k steps)

OUT=./outputs/pi05_libero_ft_my_run_$(date +%Y%m%d-%H%M%S)
LOG=./train_pi05_$(date +%Y%m%d-%H%M%S).out

# Get HF_USER from environment or use default
HF_USER=${HF_USER:-"your_hf_username"}
MASTER_PORT=${MASTER_PORT:-29515}

# Set your desired GPUs here (comma-separated), e.g., "0,1" or "2,3"
GPU_LIST="0,1"
# Avoid locking GPUs for others; we unset first, then set to your list
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASTER_PORT=$MASTER_PORT \
MUJOCO_GL=egl \
# Log the key env vars for debugging (separate command to avoid parsing torchrun args)
env | grep -E 'TIMEOUT|NCCL|MASTER_PORT'
nohup torchrun --master_port=$MASTER_PORT --nproc_per_node=4 src/lerobot/scripts/lerobot_train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.repo_id=${HF_USER}/pi05_libero_ft_my_run_custom \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --batch_size=8 \
  --steps=6000 \
  --save_freq=1000 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=-1 \
  --output_dir="$OUT" \
  > "$LOG" 2>&1 &

echo "Training started with PID: $!"
echo "Output directory: $OUT"
echo "Log file: $LOG"
echo "Monitor with: tail -f $LOG"

