# LeRobot (Private Fork)

This fork focuses on a decoupled subtask pipeline for PI05 evaluations. The goal is to extract human‑readable subtasks from evaluation rollouts without rerunning the VLA policy, and to keep outputs organized and lightweight for sharing.

## What Was Added (Summary)

1) Export subtask inputs during eval (frames + task text).
2) Offline VLM subtask generation (Qwen‑VL models).
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

### 3) Offline VLM Subtask Generation
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

### Step 2: Offline subtask generation (Qwen‑VL)
```
python examples/subtasks/generate_subtasks_offline.py \
  --inputs-dir outputs/eval/<date>/<run>/inputs/subtask_inputs \
  --output-dir outputs/eval/<date>/<run>/subtasks/qwen3 \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dtype bfloat16 \
  --temperature 0.0 \
  --max-new-tokens 16 \
  --subtask-strategy completion_check

# Alternative strategy:
# --subtask-strategy sequence_refinement
# (stores `subtask` and full `subtask_sequence` per frame)
```

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

## Dependencies (Offline VLM)
Minimal requirements for the offline scripts are listed in `requirements.txt`:
- `torch`, `torchvision`, `transformers`, `accelerate`
- `pillow`, `imageio`, `qwen-vl-utils`

## Notes
- For Qwen3‑Thinking models, outputs are cleaned to keep only the final answer.
- The completion check prompt now only uses the previous subtask (no full task).
- Sequence-refinement uses the previous predicted sequence as context to self-refine and track progress.
- Use `subtask_inputs_stride` to reduce storage or speed up VLM runs.
