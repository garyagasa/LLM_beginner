"""对比 base 模型与 SFT/DPO 模型在同一指令上的生成结果（提交实验报告用）。"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import IM_END, IM_START, format_messages
from src.lora import inject_lora, load_lora_state_dict

DEFAULT_PROMPTS = [
    "请用三句话介绍深度学习。",
    "写一首关于春天的五言绝句。",
    "Python 里 list 和 tuple 有什么区别？",
    "如何评价复旦大学管理学院的刘沛霖？",
]


def _pick_dtype(device: str) -> torch.dtype:
    if (
        device.startswith("cuda")
        and torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ):
        return torch.bfloat16
    return torch.float32


def _lora_weights_path(ckpt_dir: Path) -> Path:
    path = ckpt_dir / "lora.pt"
    if not path.exists():
        raise FileNotFoundError(f"未找到 LoRA 权重: {path}")
    return path


def _cleanup(model, device: str) -> None:
    model.cpu()
    del model
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_path: str,
    lora_ckpt: Path | None = None,
    *,
    tokenizer=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lora_r: int = 8,
    lora_alpha: float = 16,
    target_modules: list[str] | None = None,
):
    """加载 base 或 base + LoRA。"""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    dtype = _pick_dtype(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    if lora_ckpt is not None:
        inject_lora(model, target_modules=target_modules, r=lora_r, alpha=lora_alpha)
        load_lora_state_dict(
            model,
            torch.load(_lora_weights_path(lora_ckpt), map_location=device),
        )

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """单条指令生成。"""
    text = format_messages([{"role": "user", "content": prompt}])
    text += f"{IM_START}assistant\n"

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, input_ids.shape[1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=False)
    if IM_END in response:
        response = response.split(IM_END)[0]
    return response.strip()


def _collect_variants(sft_ckpt: str, dpo_ckpt: str | None) -> list[tuple[str, Path | None]]:
    variants: list[tuple[str, Path | None]] = [("Base", None)]

    sft_path = Path(sft_ckpt)
    if sft_path.exists():
        variants.append(("SFT", sft_path))
    else:
        print(f"[警告] SFT checkpoint 不存在: {sft_ckpt}，跳过 SFT")

    if dpo_ckpt:
        dpo_path = Path(dpo_ckpt)
        if dpo_path.exists():
            variants.append(("DPO", dpo_path))
        else:
            print(f"[警告] DPO checkpoint 不存在: {dpo_ckpt}，跳过 DPO")

    return variants


def _print_results(prompts: list[str], variants: list[tuple[str, Path | None]], results: dict) -> None:
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 80}")
        print(f"Prompt {i}: {prompt}")
        print("=" * 80)
        for name, _ in variants:
            print(f"\n[{name}]")
            print(results[prompt].get(name, "(未生成)"))


def main():
    p = argparse.ArgumentParser(description="对比 base vs SFT 输出")
    p.add_argument("--model", type=str, default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--sft-ckpt", type=str, default=str(ROOT / "ckpt" / "sft"))
    p.add_argument("--dpo-ckpt", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"[错误] 基座模型不存在: {args.model}\n请先运行: python data/download.py")

    target_modules = [m.strip() for m in args.target_modules.split(",")]
    variants = _collect_variants(args.sft_ckpt, args.dpo_ckpt)
    results: dict[str, dict[str, str]] = {prompt: {} for prompt in DEFAULT_PROMPTS}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, ckpt in variants:
        print(f"\n>>> 加载 {name} 模型...", flush=True)
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            ckpt,
            tokenizer=tokenizer,
            device=args.device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
        )
        for prompt in DEFAULT_PROMPTS:
            results[prompt][name] = generate(
                model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
            )
        _cleanup(model, args.device)

    print("\n" + "#" * 80)
    print("# Base / SFT / DPO 生成对比")
    print("#" * 80)
    _print_results(DEFAULT_PROMPTS, variants, results)


if __name__ == "__main__":
    main()
