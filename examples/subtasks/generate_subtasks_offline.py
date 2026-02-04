#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from huggingface_hub import hf_hub_download
except ImportError:  # Optional fallback for very old envs
    hf_hub_download = None
try:
    from safetensors import safe_open
except ImportError:  # Optional fallback for very old envs
    safe_open = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # Older Transformers
    AutoModelForVision2Seq = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older Transformers
    AutoModelForImageTextToText = None
try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:  # Older Transformers
    Qwen3VLMoeForConditionalGeneration = None


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _from_pretrained_compat(model_cls, model_name: str, dtype: torch.dtype, **extra_kwargs):
    # Keep kernels disabled for stability unless explicitly requested.
    common_kwargs = {"trust_remote_code": True, "device_map": "auto", "use_kernels": False}
    common_kwargs.update(extra_kwargs)

    # Newer Transformers versions prefer `dtype` over `torch_dtype`.
    for dtype_kw in ({"dtype": dtype}, {"torch_dtype": dtype}):
        kwargs = {**common_kwargs, **dtype_kw}
        try:
            return model_cls.from_pretrained(model_name, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument 'use_kernels'" in msg:
                kwargs.pop("use_kernels", None)
                try:
                    return model_cls.from_pretrained(model_name, **kwargs)
                except TypeError as inner_exc:
                    inner_msg = str(inner_exc)
                    if (
                        "unexpected keyword argument 'dtype'" in inner_msg
                        and "dtype" in dtype_kw
                        and "torch_dtype" not in dtype_kw
                    ):
                        continue
                    raise
            if "unexpected keyword argument 'dtype'" in msg and "dtype" in dtype_kw and "torch_dtype" not in dtype_kw:
                continue
            raise

    # Should never happen, but keep a clear failure mode.
    raise RuntimeError("Unable to load model with either `dtype` or `torch_dtype` arguments.")


def _patch_qwen3_vl_30b_moe_weights(model, model_name: str) -> int:
    """Patch MoE tensors that are transposed in some Transformers/Qwen3-VL combinations."""
    if hf_hub_download is None or safe_open is None:
        raise RuntimeError(
            "Cannot patch Qwen3-VL-30B MoE weights because `huggingface_hub` or `safetensors` is unavailable."
        )

    index_path = Path(hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json"))
    with open(index_path, "r") as f:
        index_payload = json.load(f)

    weight_map = index_payload.get("weight_map", {})
    candidate_keys = [
        key
        for key in weight_map
        if ".mlp.experts.down_proj" in key or ".mlp.experts.gate_up_proj" in key
    ]

    params = dict(model.named_parameters())
    target_keys = [key for key in candidate_keys if key in params]
    if not target_keys:
        return 0

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in target_keys:
        by_shard[weight_map[key]].append(key)

    patched = 0
    with torch.no_grad():
        for shard_name, keys in by_shard.items():
            shard_path = hf_hub_download(repo_id=model_name, filename=shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for key in keys:
                    param = params[key]
                    ckpt_tensor = shard.get_tensor(key)

                    if tuple(ckpt_tensor.shape) == tuple(param.shape):
                        fixed_tensor = ckpt_tensor
                    else:
                        transposed = ckpt_tensor.transpose(-1, -2).contiguous()
                        if tuple(transposed.shape) != tuple(param.shape):
                            raise RuntimeError(
                                f"Unexpected shape for `{key}`: ckpt={tuple(ckpt_tensor.shape)} "
                                f"model={tuple(param.shape)}"
                            )
                        fixed_tensor = transposed

                    param.data.copy_(fixed_tensor.to(device=param.device, dtype=param.dtype))
                    patched += 1
    return patched


def _strip_thinking(text: str) -> str:
    lowered = text.lower()
    if "<think>" in lowered:
        start = lowered.find("<think>")
        end = lowered.find("</think>", start + 7)
        if end != -1:
            return text[:start] + text[end + 8 :]
    return text


def _extract_tag(text: str, tag: str) -> str | None:
    lowered = text.lower()
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = lowered.find(open_tag)
    end = lowered.find(close_tag)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(open_tag) : end].strip()


def _extract_subtask_sentence(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text.strip()
    blacklist = ("got it", "let's", "task is", "the task is", "task:", "image", "i will", "i can")
    for sentence in reversed(sentences):
        lowered = sentence.lower()
        if any(term in lowered for term in blacklist):
            continue
        return sentence
    return sentences[-1]


def _clean_response(text: str) -> str:
    text = _strip_thinking(text).strip()
    lowered = text.lower()
    tagged = _extract_tag(text, "subtask")
    if tagged:
        return tagged
    if "final:" in lowered:
        text = text[lowered.rfind("final:") + len("final:") :].strip()
    if text.lower().startswith("assistant"):
        text = text.split(":", 1)[-1].strip()
    text = text.strip().split(";", 1)[0].strip()
    text = _extract_subtask_sentence(text)
    cleaned = text.strip().split("\n", 1)[0].strip()
    lowered = cleaned.lower()
    if "task" in lowered and ("task is" in lowered or "the task" in lowered or "task:" in lowered):
        return ""
    return cleaned


def _build_prompt(task: str, prev_subtask: str, *, model_name: str) -> str:
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    prev_text = prev_subtask.strip() if prev_subtask else "none"
    prefix = "<image>\n" if "qwen3-vl" in model_name.lower() else ""
    return (
        f"{prefix}You are a robot. From the task and the current image, output the next subtask.\n"
        "Use 3-8 words. Use an imperative verb phrase.\n"
        "If nothing changed since the previous step, repeat the previous subtask.\n"
        "If the previous subtask is complete, update it to the next one.\n"
        "Do not restate the task. Output only the subtask in this XML tag: <subtask>...</subtask>.\n"
        "\nExample:\n"
        "Task: put the red block in the bin\n"
        "Prev subtask: move to red block\n"
        "Next subtask: <subtask>grasp red block</subtask>\n"
        "\n"
        f"Task: {cleaned_text}\n"
        f"Prev subtask: {prev_text}\n"
        "Next subtask: <subtask>"
    )


def _build_completion_prompt(task: str, prev_subtask: str, *, model_name: str) -> str:
    prev_text = prev_subtask.strip() if prev_subtask else "none"
    prefix = "<image>\n" if "qwen3-vl" in model_name.lower() else ""
    return (
        f"{prefix}You are a robot. Look at the current image and decide if the previous subtask is completed.\n"
        "Reply with only one word: yes or no.\n"
        f"Prev subtask: {prev_text}\n"
        "Completed:"
    )


def _select_image(frame: dict, image_key: str | None, *, base_dir: Path) -> tuple[Image.Image, str]:
    images = frame.get("images", {})
    if not images:
        raise ValueError("Frame has no images.")
    if image_key and image_key in images:
        chosen_key = image_key
    else:
        chosen_key = sorted(images.keys())[0]
    image_path = Path(images[chosen_key])
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    return Image.open(image_path).convert("RGB"), chosen_key


def _prepare_inputs(processor, image: Image.Image, text: str, *, use_qwen_chat: bool):
    if use_qwen_chat and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=chat_text, images=image, return_tensors="pt", padding=True)
    return processor(text=text, images=image, return_tensors="pt", padding=True)


def _generate_text(
    model,
    processor,
    image: Image.Image,
    text: str,
    *,
    temperature: float,
    max_new_tokens: int | None,
    min_new_tokens: int,
    disable_eos: bool,
    clean_output: bool,
):
    inputs = _prepare_inputs(
        processor,
        image,
        text,
        use_qwen_chat="qwen" in model.config.model_type if hasattr(model, "config") else False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = {"min_new_tokens": min_new_tokens, "do_sample": temperature > 0}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    if disable_eos:
        gen_kwargs["eos_token_id"] = None
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated[0, input_len:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    if clean_output:
        return _clean_response(text)
    return text.strip()


def generate_subtasks_for_episode(
    episode_payload: dict,
    *,
    model,
    processor,
    model_name: str,
    image_key: str | None,
    temperature: float,
    max_new_tokens: int | None,
    use_completion_check: bool,
    base_dir: Path,
    min_new_tokens: int,
    disable_eos: bool,
    clean_output: bool,
) -> dict:
    frames = episode_payload.get("frames", [])
    task_text = episode_payload.get("task", "")
    prev_subtask = ""
    outputs = []
    for frame in frames:
        image, chosen_key = _select_image(frame, image_key, base_dir=base_dir)
        prompt = _build_prompt(task_text, prev_subtask, model_name=model_name)
        candidate = _generate_text(
            model,
            processor,
            image,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            disable_eos=disable_eos,
            clean_output=clean_output,
        )
        final = candidate
        if use_completion_check and prev_subtask:
            completion_prompt = _build_completion_prompt(task_text, prev_subtask, model_name=model_name)
            completed = _generate_text(
                model,
                processor,
                image,
                completion_prompt,
                temperature=0.0,
                max_new_tokens=3,
                min_new_tokens=1,
                disable_eos=False,
                clean_output=True,
            )
            if not completed.lower().startswith("y"):
                final = prev_subtask
        prev_subtask = final
        outputs.append(
            {
                "step": frame.get("step"),
                "image_key": chosen_key,
                "subtask": final,
            }
        )
    episode_payload["subtasks"] = outputs
    return episode_payload


def _iter_task_dirs(inputs_dir: Path) -> list[Path]:
    if any(p.name.startswith("episode_") and p.suffix == ".json" for p in inputs_dir.iterdir()):
        return [inputs_dir]
    return [p for p in inputs_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-max-new-tokens", action="store_true")
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--disable-eos", action="store_true")
    parser.add_argument("--no-clean-output", action="store_true")
    parser.add_argument("--no-completion-check", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = _resolve_dtype(args.dtype)

    model_name = args.model
    max_new_tokens = None if args.no_max_new_tokens else args.max_new_tokens
    clean_output = not args.no_clean_output
    if "qwen3-vl-30b-a3b" in model_name.lower() and Qwen3VLMoeForConditionalGeneration is not None:
        try:
            model = _from_pretrained_compat(
                Qwen3VLMoeForConditionalGeneration,
                model_name,
                torch_dtype,
                ignore_mismatched_sizes=False,
            )
        except RuntimeError as exc:
            if "ignore_mismatched_sizes" not in str(exc):
                raise
            # Known HF/Qwen3-VL mismatch in some builds: load, then patch MoE tensors from checkpoint.
            model = _from_pretrained_compat(
                Qwen3VLMoeForConditionalGeneration,
                model_name,
                torch_dtype,
                ignore_mismatched_sizes=True,
            )
            patched = _patch_qwen3_vl_30b_moe_weights(model, model_name)
            print(f"Patched {patched} Qwen3-VL MoE tensors from checkpoint.")
    elif AutoModelForImageTextToText is not None:
        model = _from_pretrained_compat(AutoModelForImageTextToText, model_name, torch_dtype)
    elif AutoModelForVision2Seq is not None:
        model = _from_pretrained_compat(AutoModelForVision2Seq, model_name, torch_dtype)
    else:
        raise RuntimeError(
            "Your Transformers version is too old for Qwen3-VL. "
            "Install a newer Transformers (or from source) that provides "
            "AutoModelForImageTextToText or AutoModelForVision2Seq."
        )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    task_dirs = _iter_task_dirs(args.inputs_dir)
    for task_dir in task_dirs:
        metadata = {}
        metadata_path = task_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        episodes = sorted(task_dir.glob("episode_*.json"))
        output_task_dir = args.output_dir / task_dir.name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        if metadata:
            with open(output_task_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        for episode_path in episodes:
            with open(episode_path, "r") as f:
                episode_payload = json.load(f)
            enriched = generate_subtasks_for_episode(
                episode_payload,
                model=model,
                processor=processor,
                model_name=model_name,
                image_key=args.image_key,
                temperature=args.temperature,
                max_new_tokens=max_new_tokens,
                use_completion_check=not args.no_completion_check,
                base_dir=task_dir,
                min_new_tokens=args.min_new_tokens,
                disable_eos=args.disable_eos,
                clean_output=clean_output,
            )
            output_path = output_task_dir / episode_path.name
            with open(output_path, "w") as f:
                json.dump(enriched, f, indent=2)


if __name__ == "__main__":
    main()
