"""任务二：Causal multi-head attention + KV cache。

接口约定（eval/run.py 会自动检测）：
  - KV cache 等价性：开/不开 cache 的 logits 一致（误差 < 1e-4）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb


class CausalMultiHeadAttention(nn.Module):
    """带 RoPE 和因果 mask 的多头注意力，支持 KV cache。"""

    def __init__(self, d_model: int, n_heads: int, max_len: int = 2048):
        """初始化。

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            max_len: 最大序列长度（用于 RoPE 预计算）
        """
        # TODO: 1. 初始化 Q/K/V/O 线性层
        # TODO: 2. 预计算 RoPE freqs_cis
        super().__init__()
        raise NotImplementedError("TODO: 实现 CausalMultiHeadAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_cache: bool = False,
    ):
        """前向传播。

        Args:
            x: (B, T, d_model)  当前输入
            kv_cache: 可选的 (K_cache, V_cache)，形状 (B, H, T_past, D)
            return_cache: 是否返回更新后的 KV cache

        Returns:
            若 return_cache=False: output (B, T, d_model)
            若 return_cache=True:  (output, (new_K_cache, new_V_cache))
        """
        # TODO: 1. 线性投影 Q/K/V
        # TODO: 2. 施加 RoPE 到 Q 和 K
        # TODO: 3. 如果有 kv_cache，拼接历史 K/V
        # TODO: 4. 生成 causal mask（下三角）
        # TODO: 5. scaled_dot_product_attention
        # TODO: 6. O 投影
        # TODO: 7. 按 return_cache 决定返回值
        raise NotImplementedError("TODO: 实现 CausalMultiHeadAttention.forward")
