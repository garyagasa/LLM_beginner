"""任务二：自回归采样策略（greedy / top-k / top-p / temperature）。"""

from typing import List
import torch
import torch.nn.functional as F


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """贪心采样：选概率最高的 token。

    Args:
        logits: (B, vocab_size)

    Returns:
        (B,) 选中的 token ids
    """
    # TODO: 取 argmax
    raise NotImplementedError("TODO: 实现 sample_greedy")


def sample_top_k(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """Top-k 采样。

    Args:
        logits: (B, vocab_size)
        k: 保留 top-k
        temperature: 温度系数

    Returns:
        (B,) 采样得到的 token ids
    """
    # TODO: 1. logits /= temperature
    # TODO: 2. 取 top-k，其余设为 -inf
    # TODO: 3. softmax + multinomial 采样
    raise NotImplementedError("TODO: 实现 sample_top_k")


def sample_top_p(logits: torch.Tensor, top_p: float, temperature: float = 1.0) -> torch.Tensor:
    """Nucleus (top-p) 采样：保留累积概率 >= top_p 的 token。

    Args:
        logits: (B, vocab_size)
        top_p: 累积概率阈值
        temperature: 温度系数

    Returns:
        (B,) 采样得到的 token ids
    """
    # TODO: 1. logits /= temperature
    # TODO: 2. 按概率从大到小排序
    # TODO: 3. 取累积概率 <= top_p 的 token，其余设为 -inf
    # TODO: 4. softmax + multinomial 采样
    raise NotImplementedError("TODO: 实现 sample_top_p")
