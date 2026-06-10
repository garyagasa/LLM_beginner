"""M1 + M2：手写 scaled-dot-product attention 与 MultiHeadAttention。

---- 接口约定（eval/run.py 自检依赖）----
  scaled_dot_product_attention(Q, K, V, mask=None)
    Q/K/V 形状: (B, H, T, D)   B=Batch, H=Heads, T=seq_len, D=head_dim
    mask:    True=被屏蔽的位置（填 -inf）
    返回:    (B, H, T, D)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  M1：缩放点积注意力（scaled dot-product attention）
# ================================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """缩放点积注意力。

    Args:
        Q: Query  (B, H, T, d_k)
        K: Key    (B, H, T, d_k)
        V: Value  (B, H, T, d_k)
        mask: 可选，True 的位置被屏蔽（填 -inf）。

    Returns:
        (B, H, T, d_k)

    实现步骤：
        1. scores = Q @ K^T / sqrt(d_k)        ← 点积 + 缩放
        2. if mask: scores.masked_fill(mask, -inf)  ← mask 屏蔽
        3. attn_weights = softmax(scores, dim=-1)   ← 归一化
        4. output = attn_weights @ V                 ← 加权求和
    """
    # ---- 你的代码开始 ----
    # TODO: 实现上述 4 步；参考上面注释的步骤

    d_k = Q.size(-1)                                             # step 0 取出 d_k
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)            # step 1: Q @ K^T / sqrt(d_k)，形状 (B, H, T, T)
    if mask is not None:                                         # step 2: mask 处理
        scores = scores.masked_fill(mask,float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)                        # step 3: softmax(dim=-1)
    output = attn_weights @ V                                     # step 4: 加权求和
    # ---- 你的代码结束 ----
    return output


# ================================================================
#  M2：多头注意力（Multi-Head Attention）
# ================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力：Q/K/V 投影 → 分头 → 缩放点积 → 合并 → 输出投影。

    不应调用 nn.MultiheadAttention 或任何高层封装。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型总维度
            n_heads: 注意力头数（d_model 必须能被 n_heads 整除）
            dropout: attention 输出的 dropout 概率
        """
        super().__init__()
        # TODO: 检查 d_model % n_heads == 0，否则报错
        if not d_model % n_heads == 0:
            raise ValueError("d_model must be divisible by n_heads")
        # TODO: 保存 d_model, n_heads, d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # TODO: 创建 4 个线性投影层（bias=False）：
        #   self.W_q / self.W_k / self.W_v  → Q/K/V 投影 (d_model → d_model)
        #   self.W_o                         → 输出投影 (d_model → d_model)
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model, bias = False)
        # TODO: 创建 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x:    (B, T, d_model)
            mask: 可选 attention mask，True = 屏蔽

        Returns:
            (B, T, d_model)

        实现步骤：
            1. Q/K/V 投影: x → Linear → (B, T, d_model)
            2. 分头: reshape → (B, T, H, d_k) → transpose → (B, H, T, d_k)
            3. 调用 scaled_dot_product_attention(Q, K, V, mask)
            4. 合并头: transpose → (B, T, H, d_k) → reshape → (B, T, d_model)
            5. 输出投影 + dropout
        """
        B, T, _ = x.shape
        # ---- 你的代码开始 ----
        # TODO: 实现上述 5 步
        # Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # (B, H, T, d_k)
        # K = ...
        # V = ...
        # attn_out = scaled_dot_product_attention(Q, K, V, mask)                # (B, H, T, d_k)
        # attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        # output = self.W_o(attn_out)
        # return self.dropout(output)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(attn_out)
        return self.dropout(output)
        # ---- 你的代码结束 ----
