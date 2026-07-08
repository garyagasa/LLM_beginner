"""对比 base 模型与 SFT/DPO 模型在同一指令上的生成结果（提交实验报告用）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PROMPTS = [
    "请用三句话介绍深度学习。",
    "写一首关于春天的五言绝句。",
    "Python 里 list 和 tuple 有什么区别？",
]


def load_model_and_tokenizer(model_path: str, lora_ckpt: Path | None = None):
    """加载 base 或 base + LoRA。"""
    # TODO: AutoModelForCausalLM + AutoTokenizer
    # TODO: 若 lora_ckpt 给定，inject_lora 后 load_lora_state_dict
    raise NotImplementedError("TODO: 实现 load_model_and_tokenizer")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """单条指令生成。"""
    # TODO: format_messages([{"role": "user", "content": prompt}])
    # TODO: model.generate(...) 并 decode
    raise NotImplementedError("TODO: 实现 generate")


def main():
    p = argparse.ArgumentParser(description="对比 base vs SFT 输出")
    p.add_argument("--model", type=str, default=str(ROOT / "models" / "Qwen2.5-0.5B"))
    p.add_argument("--sft-ckpt", type=str, default=str(ROOT / "ckpt" / "sft"))
    p.add_argument("--dpo-ckpt", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    # TODO: 对 DEFAULT_PROMPTS 分别跑 base / SFT / DPO，打印对比表格
    print("TODO: 实现 compare.py main —— 跑完 SFT 后用于手动确认 sft_vs_base 测试")


if __name__ == "__main__":
    main()
