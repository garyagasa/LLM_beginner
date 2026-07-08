"""M2：Qwen chat template + loss masking。

---- 接口约定（eval/run.py 自检依赖）----
  format_messages(messages: List[dict]) -> str
    将 [{"role": "user"|"assistant"|"system", "content": "..."}] 格式化为 Qwen 对话字符串。
  build_labels(input_ids, messages) -> labels
    与 input_ids 同 shape；user/system/模板控制符位置为 -100，assistant 内容为真实 token id。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

# Qwen2.5 官方 special tokens（tokenizer 里也有，这里便于对照）
IM_START = "<|im_start|>"
IM_END = "<|" + "im_end" + "|>"  # Qwen2.5 对话结束符

DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"


def format_messages(messages: List[dict]) -> str:
    """应用 Qwen chat template，拼接多轮对话。"""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return "".join(parts)


def _format_turn(msg: dict) -> str:
    return f"{IM_START}{msg['role']}\n{msg['content']}{IM_END}\n"


def _get_tokenizer(tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(DEFAULT_TOKENIZER_PATH))


def build_labels(
    input_ids: torch.Tensor,
    messages: List[dict],
    tokenizer=None,
    max_length: int | None = None,
) -> torch.Tensor:
    """构造 SFT 用的 labels：只对 assistant turn 的正文计算 loss。"""
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)

    labels = torch.full_like(input_ids, -100)
    text = format_messages(messages)
    tokenizer = _get_tokenizer(tokenizer)

    encode_kwargs: dict = {
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        encode_kwargs["truncation"] = True
        encode_kwargs["max_length"] = max_length

    encoded = tokenizer(text, **encode_kwargs)
    offsets = encoded.offset_mapping[0]
    seq_len = input_ids.size(0)

    pos = 0
    for msg in messages:
        turn = _format_turn(msg)
        turn_start = pos
        turn_end = pos + len(turn)
        pos = turn_end

        if msg["role"] != "assistant":
            continue

        header_len = len(f"{IM_START}{msg['role']}\n")
        footer_len = len(f"{IM_END}\n")
        content_start = turn_start + header_len
        content_end = turn_end - footer_len

        for i, span in enumerate(offsets):
            if i >= seq_len:
                break
            if span is None:
                continue
            char_start, char_end = span
            if char_start >= content_start and char_end <= content_end:
                labels[i] = input_ids[i]

    return labels
