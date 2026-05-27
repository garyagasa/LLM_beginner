"""任务一：实现 scaled dot-product attention + MultiHeadAttention。

接口约定（eval/run.py 会自动检测）：
  - scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor
    Q/K/V 形状 (B, H, T, D)，mask 中 True = 被屏蔽
  - class MultiHeadAttention(nn.Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """手写 scaled dot-product attention。

    Args:
        Q: (B, H, T_q, D)  queries
        K: (B, H, T_k, D)  keys
        V: (B, H, T_v, D)  values  (T_k == T_v)
        mask: (B, 1, T_q, T_k) or broadcastable; True = 被屏蔽

    Returns:
        output: (B, H, T_q, D)
    """
    # TODO: 实现 scaled dot-product attention
    # 1. d_k = Q.size(-1)
    # 2. scores = Q @ K^T / sqrt(d_k)
    # 3. if mask is not None: scores = scores.masked_fill(mask, float('-inf'))
    # 4. attn_weights = F.softmax(scores, dim=-1)
    # 5. output = attn_weights @ V
    raise NotImplementedError("TODO: 实现 scaled_dot_product_attention")


class MultiHeadAttention(nn.Module):
    """多头注意力层。"""

    def __init__(self, d_model: int, n_heads: int):
        """初始化。

        Args:
            d_model: 模型总维度
            n_heads: 注意力头数
        """
        # TODO: 检查 d_model % n_heads == 0
        # TODO: 初始化 Q/K/V/O 四个线性层
        super().__init__()
        raise NotImplementedError("TODO: 实现 MultiHeadAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, T, d_model)  输入
            mask: attention mask

        Returns:
            output: (B, T, d_model)
        """
        # TODO: 1. 线性投影 Q/K/V
        # TODO: 2. reshape 为 (B, H, T, D)
        # TODO: 3. 调用 scaled_dot_product_attention
        # TODO: 4. reshape 回 (B, T, d_model)
        # TODO: 5. O 投影
        raise NotImplementedError("TODO: 实现 MultiHeadAttention.forward")
