"""任务二：RoPE（Rotary Position Embedding）实现。"""

import torch


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0):
    """预计算旋转频率的复数表示。

    Args:
        dim: head 维度
        max_len: 最大序列长度
        theta: base theta

    Returns:
        freqs_cis: (max_len, dim // 2) 的复数 tensor
    """
    # TODO: 1. 计算 freq = 1 / (theta ** (torch.arange(0, dim, 2) / dim))
    # TODO: 2. 生成 position = torch.arange(max_len)
    # TODO: 3. 外积得到 (max_len, dim // 2) 的角度
    # TODO: 4. 用 torch.polar 转为复数
    raise NotImplementedError("TODO: 实现 precompute_freqs_cis")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """对输入 x 施加 RoPE。

    Args:
        x: (B, H, T, D) 的 query 或 key
        freqs_cis: 预计算的复数频率

    Returns:
        施加 RoPE 后的 tensor，形状同 x
    """
    # TODO: 1. 将 x 转为复数形式（两两一组）
    # TODO: 2. 乘以 freqs_cis
    # TODO: 3. 转回实数形式
    raise NotImplementedError("TODO: 实现 apply_rotary_emb")
