"""M2：TransformerBlock = MultiHeadAttention + FFN + residual + LayerNorm。

推荐使用 Pre-norm 风格（更稳定，GPT-2 / Llama 约定）：
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

S2 消融：可通过 use_residual / use_layernorm 开关对比训练收敛性。
"""
import torch
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
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


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

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff:    FFN 中间层维度（通常是 d_model × 4）
            dropout: dropout 概率
            use_residual: 是否使用 residual 连接（S2 消融）
            use_layernorm: 是否使用 LayerNorm（S2 消融）
        """
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            x:    (B, T, d_model)
            mask: 可选 attention mask，True = 屏蔽
            rotary_emb: 可选 RoPE 模块

        Returns:
            (B, T, d_model)
        """
        attn_out = self.attention(self.norm1(x), mask, rotary_emb=rotary_emb)
        x = x + attn_out if self.use_residual else attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out if self.use_residual else ffn_out
        return x
