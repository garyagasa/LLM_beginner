"""任务一：Transformer block（attention + FFN + residual + LayerNorm）。"""

import torch.nn as nn
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """一个标准的 Transformer encoder block。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """初始化。

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: FFN 中间层维度
            dropout: dropout 概率
        """
        # TODO: 初始化 self.attention = MultiHeadAttention(...)
        # TODO: 初始化 self.norm1, self.norm2 = nn.LayerNorm(d_model)
        # TODO: 初始化 FFN: Linear -> ReLU/GeLU -> Linear -> Dropout
        # TODO: 初始化 dropout
        super().__init__()
        raise NotImplementedError("TODO: 实现 TransformerBlock.__init__")

    def forward(self, x, mask=None):
        """前向传播。

        Args:
            x: (B, T, d_model)
            mask: attention mask

        Returns:
            output: (B, T, d_model)
        """
        # TODO: 1. attn_out = self.attention(self.norm1(x), mask)
        # TODO: 2. x = x + self.dropout(attn_out)
        # TODO: 3. ffn_out = self.ffn(self.norm2(x))
        # TODO: 4. x = x + self.dropout(ffn_out)
        raise NotImplementedError("TODO: 实现 TransformerBlock.forward")
