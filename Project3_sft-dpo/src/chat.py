"""M2：Qwen chat template + loss masking。

---- 接口约定（eval/run.py 自检依赖）----
  format_messages(messages: List[dict]) -> str
    将 [{"role": "user"|"assistant"|"system", "content": "..."}] 格式化为 Qwen 对话字符串。
  build_labels(input_ids, messages) -> labels
    与 input_ids 同 shape；user/system/模板控制符位置为 -100，assistant 内容为真实 token id。
"""

from __future__ import annotations

from typing import List

import torch

# Qwen2.5 官方 special tokens（tokenizer 里也有，这里便于对照）
IM_START = "<|im_start|>"
IM_END = "<|" + "im_end" + "|>"  # Qwen2.5 对话结束符


def format_messages(messages: List[dict]) -> str:
    """应用 Qwen chat template，拼接多轮对话。

    格式示例（单轮）：
        <|im_start|>user
        你好
        <|im_start|>assistant
        你好！很高兴见到你。

    Args:
        messages: 按时间顺序的 role/content 列表

    Returns:
        完整 prompt 字符串（含最后一个 assistant turn；训练时通常保留完整对话）

    实现步骤：
        1. 遍历 messages，按 role 拼接 IM_START + role + "\\n" + content + IM_END + "\\n"
        2. 若无 system 消息，可选在开头加默认 system（MOSS 数据通常已有 system）
        3. 参考 transformers 的 Qwen2Tokenizer.apply_chat_template(..., tokenize=False)
    """
    # ---- 你的代码开始 ----
    # TODO: 拼接 Qwen chat template
    raise NotImplementedError("TODO: 实现 format_messages")
    # ---- 你的代码结束 ----


def build_labels(input_ids: torch.Tensor, messages: List[dict]) -> torch.Tensor:
    """构造 SFT 用的 labels：只对 assistant turn 计算 loss。

    Args:
        input_ids: shape (seq_len,) 或 (1, seq_len)，tokenize(format_messages(messages)) 的结果
        messages:  与 format_messages 相同的 messages 列表

    Returns:
        labels: 与 input_ids 同 shape；不参与 loss 的位置填 -100

    实现步骤：
        1. 对每个 turn 单独 tokenize「从该 turn 开头到 turn 结束」的片段，定位在完整序列中的 span
        2. user / system turn 及 IM_START/IM_END 控制符 → labels 填 -100
        3. assistant turn 的正文 token → 保留 input_ids 对应 id（可多轮 assistant 都参与）
        4. 推荐：逐 turn 用 tokenizer 对齐，避免纯字符串 find 在中文/空格下错位

    提示：
        - eval 期望 mask 比例在 20%-90%（user/system 全 -100）
        - labels 通常右移一位做 next-token prediction，也可  eval 只检查 shape 与 mask 比例
    """
    # ---- 你的代码开始 ----
    # TODO: 构造 labels，非 assistant 位置填 -100
    raise NotImplementedError("TODO: 实现 build_labels")
    # ---- 你的代码结束 ----
