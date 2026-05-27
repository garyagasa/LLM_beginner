"""任务三：chat template 与 loss masking。

接口约定（eval/run.py 会自动检测）：
  - format_messages(messages: List[dict]) -> str  应用 Qwen chat template
  - build_labels(input_ids, messages) -> labels  做 loss masking
"""

from typing import List, Dict
import torch


def format_messages(messages: List[Dict[str, str]]) -> str:
    """将对话消息列表格式化为 Qwen chat template 字符串。

    Args:
        messages: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]

    Returns:
        str: 格式化后的 prompt
    """
    # TODO: 应用 Qwen2.5 chat template
    # 格式：<|im_start|>system\n{content}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n...
    raise NotImplementedError("TODO: 实现 format_messages")


def build_labels(
    input_ids: torch.Tensor,
    messages: List[Dict[str, str]],
) -> torch.Tensor:
    """构建 loss masking 的 labels。

    user/system 部分标签设为 -100（不被计算 loss），
    只有 assistant 部分参与 loss 计算。

    Args:
        input_ids: (T,)  已编码的完整对话 token ids
        messages: 对应的对话消息列表

    Returns:
        labels: (T,)  同 input_ids，但在 user/system 位置为 -100
    """
    # TODO: 1. 对每个消息，找到其在 input_ids 中的起止位置
    # TODO: 2. 对 role == "user" 或 "system" 的消息，对应 token 位置设为 -100
    # TODO: 3. 对 role == "assistant" 的消息，保留原始 token id
    raise NotImplementedError("TODO: 实现 build_labels")
