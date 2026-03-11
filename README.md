# LeRobot (Private Fork)

This fork focuses on a decoupled subtask pipeline for PI05 evaluations. The goal is to extract human‑readable subtasks from evaluation rollouts without rerunning the VLA policy, and to keep outputs organized and lightweight for sharing.

## What Was Added (Summary)

1) Export subtask inputs during eval (frames + task text).
2) Offline subtask generation with either VLM (Qwen‑VL) or MLLM (OpenAI GPT).
3) Two‑stage subtask logic (generate candidate, then completion check).
4) Video annotation with per‑frame subtasks.
5) Output/log reorganization helpers.

## Detailed Additions

### 1) Decoupled Subtask Generation
Instead of generating subtasks inside the policy loop, this workflow exports frames + task text during eval, then runs an external VLM offline.

Benefits:
- No dependency on the model version used for PI05 inference.
- Easy to swap VLMs (Qwen3, Qwen3‑Thinking, etc.).
- Fast iteration on prompts and filtering without touching rollouts.

### 2) Subtask Input Export During Eval
`lerobot-eval` now supports exporting per‑frame inputs for offline generation.

Config flags (`EvalConfig`):
- `eval.export_subtask_inputs`: enable export
- `eval.subtask_inputs_stride`: save every N frames

Exported data:
- PNG frames per step
- Task text per episode
- JSON index of frames and paths

### 3) Offline Subtask Generation (VLM or MLLM)
New script to generate subtasks from stored frames:
- `examples/subtasks/generate_subtasks_offline.py`

Key behavior:
- Prompts for short imperative subtasks.
- Optional completion check (yes/no) to keep previous subtask if incomplete.
- Optional sequence-refinement strategy that predicts the full remaining subtask plan per frame and keeps the first item as the main subtask.
- Cleans up model outputs (removes thinking blocks, trims acknowledgments).

### 4) Video Annotation
New script to burn subtasks onto videos:
- `examples/subtasks/annotate_videos_with_subtasks.py`

It creates `.srt` subtitles and uses `ffmpeg` to render per‑frame labels.

### 5) Output Organization
New script to standardize eval outputs and logs:
- `examples/subtasks/organize_run_outputs.py`

It groups `subtask_inputs`, `videos_*`, and `subtasks_*` into consistent folders and moves logs to a central location.

## Key Files Modified

Evaluation + export:
- `src/lerobot/configs/default.py`
- `src/lerobot/scripts/lerobot_eval.py`

PI05 subtask support (prompting + external VLM hooks):
- `src/lerobot/policies/pi05/configuration_pi05.py`
- `src/lerobot/policies/pi05/modeling_pi05.py`
- `src/lerobot/policies/pi05/processor_pi05.py`

Scripts:
- `examples/subtasks/run_decoupled_subtasks.sh`
- `examples/subtasks/generate_subtasks_offline.py`
- `examples/subtasks/annotate_videos_with_subtasks.py`
- `examples/subtasks/organize_run_outputs.py`

## End‑to‑End Usage

### Step 1: Export subtask inputs (PI05 eval)
```
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --eval.n_episodes=5 \
  --eval.batch_size=5 \
  --eval.export_subtask_inputs=true \
  --eval.subtask_inputs_stride=1 \
  --policy.device=cuda
```

### Step 2: Offline subtask generation

Hugging Face VLM (existing flow, unchanged):
```
python examples/subtasks/generate_subtasks_offline.py \
  --backend hf \
  --inputs-dir outputs/eval/<date>/<run>/inputs/subtask_inputs \
  --output-dir outputs/eval/<date>/<run>/subtasks/qwen3 \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dtype bfloat16 \
  --temperature 0.0 \
  --max-new-tokens 16 \
  --subtask-strategy completion_check

# Alternative strategy:
# --subtask-strategy pick_list
# (stores `subtask` and full `subtask_sequence` per frame)
# GPT-5.2-friendly strategy:
# --subtask-strategy pick_list_monotonic
# (generates a fixed plan once, then tracks a monotonic step index)
# Binary transition baseline:
# --subtask-strategy plan_once_binary_advance
# (generates a fixed plan once, then asks only yes/no to advance)
# Global alignment baseline:
# --subtask-strategy plan_once_global_alignment
# (generates a fixed plan once, scores anchor frames, then globally aligns)
```

OpenAI MLLM (new option):
```
python examples/subtasks/generate_subtasks_offline.py \
  --backend openai \
  --inputs-dir outputs/eval/<date>/<run>/inputs/subtask_inputs \
  --output-dir outputs/eval/<date>/<run>/subtasks/gpt52 \
  --model gpt-5.2 \
  --openai-image-detail high \
  --openai-reasoning-effort low \
  --temperature 0.0 \
  --max-new-tokens 16 \
  --max-episodes-per-task 1 \
  --subtask-strategy pick_list_monotonic

# Alternative offline baseline:
# --subtask-strategy plan_once_global_alignment
# --global-alignment-num-anchors 8
```

Notes:
- `--backend` defaults to `hf`, so current VLM behavior stays the default.
- For OpenAI backend, set `OPENAI_API_KEY` in your environment.
- For a low-cost smoke test, add `--max-episodes-per-task 1` to process one sample per task.
- `pick_list_monotonic` is the recommended GPT-5.2 mode: it plans once, then tracks only the current step index with no backward jumps.
- `plan_once_binary_advance` is a simpler comparison baseline: it plans once, then makes only a binary advance/no-advance decision per frame.
- `plan_once_global_alignment` is a trajectory-level baseline: it plans once, scores a sparse set of anchor frames, then fits a globally monotonic timeline.

### Step 3: Annotate videos
```
python examples/subtasks/annotate_videos_with_subtasks.py \
  --subtasks-root outputs/eval/<date>/<run>/subtasks/qwen3 \
  --output-root outputs/eval/<date>/<run>/videos/with_subtasks \
  --overwrite
```

### Step 4: Organize outputs + logs
```
python examples/subtasks/organize_run_outputs.py \
  --run-dir outputs/eval/<date>/<run>
```

## Output Layout (After Organization)
- `outputs/eval/<date>/<run>/inputs/subtask_inputs/`
- `outputs/eval/<date>/<run>/videos/<suite>/`
- `outputs/eval/<date>/<run>/videos/with_subtasks/`
- `outputs/eval/<date>/<run>/subtasks/<variant>/`
- `outputs/logs/` (moved `nohup*.log`, `train_*.out`)

## Dependencies (Offline Subtask Generation)
Minimal requirements for the offline scripts are listed in `requirements.txt`:
- `torch`, `torchvision`, `transformers`, `accelerate`
- `pillow`, `imageio`, `qwen-vl-utils`, `openai`

## Notes
- For Qwen3‑Thinking models, outputs are cleaned to keep only the final answer.
- In `completion_check`, the previous subtask is checked first and a new subtask is generated only if completion is `yes`.
- `pick_list` generates a full list at t=0, then picks the current subtask from that fixed list at each step.
- `pick_list_monotonic` generates a full list at t=0, then tracks a 1-based subtask index that can only stay the same or advance by one.
- `plan_once_binary_advance` generates a full list at t=0, then keeps the current step unless the model answers `yes` to a binary completion check for that step.
- `plan_once_global_alignment` generates a full list at t=0, predicts subtask indices only on sparse anchor frames, then uses monotonic global alignment and interpolation to label the full trajectory.
- Use `subtask_inputs_stride` to reduce storage or speed up VLM runs.
