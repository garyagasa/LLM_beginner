"""M2：TransformerBlock = MultiHeadAttention + FFN + residual + LayerNorm。

推荐使用 Pre-norm 风格（更稳定，GPT-2 / Llama 约定）：
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
"""

import torch.nn as nn

from src.attention import MultiHeadAttention


# ================================================================
#  FeedForward（Position-wise FFN）
# ================================================================

class FeedForward(nn.Module):
    """两层全连接 + GELU 激活。

    Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model) → Dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: 创建 Sequential：
        #   nn.Linear(d_model, d_ff)
        #   nn.GELU()
        #   nn.Dropout(dropout)
        #   nn.Linear(d_ff, d_model)
        #   nn.Dropout(dropout)
        raise NotImplementedError("TODO: 实现 FeedForward.__init__ — 约 5 行")

    def forward(self, x):
        # TODO: return self.net(x)
        raise NotImplementedError("TODO: 实现 FeedForward.forward — 1 行")


# ================================================================
#  TransformerBlock
# ================================================================

class TransformerBlock(nn.Module):
    """一个 Transformer 层（encoder block）。

    结构（Pre-norm）：
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    无 causal mask —— 这是 encoder block，所有 token 互相可见。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff:    FFN 中间层维度（通常是 d_model × 4）
            dropout: dropout 概率
        """
        super().__init__()
        # TODO: 创建子模块
        #   self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        #   self.ffn = FeedForward(d_model, d_ff, dropout)
        #   self.norm1 = nn.LayerNorm(d_model)    ← attention 前的 LayerNorm
        #   self.norm2 = nn.LayerNorm(d_model)    ← FFN 前的 LayerNorm
        raise NotImplementedError("TODO: 实现 TransformerBlock.__init__ — 约 4 行")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x:    (B, T, d_model)
            mask: 可选 attention mask，True = 屏蔽

        Returns:
            (B, T, d_model)

        实现步骤（Pre-norm）：
            1. x = x + attention(norm1(x), mask)
            2. x = x + ffn(norm2(x))
        """
        # ---- 你的代码开始 ----
        # TODO: 实现 Pre-norm 的两步 residual
        # x = x + self.attention(self.norm1(x), mask)
        # x = x + self.ffn(self.norm2(x))
        raise NotImplementedError("TODO: 实现 TransformerBlock.forward — 约 2 行")
        # ---- 你的代码结束 ----
        return x
