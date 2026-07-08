"""Causal multi-head self-attention with optional KV cache."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=block_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
        return_cache: bool = False,
    ):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, T, hs)

        q, k = self.rope.apply_rotary(q, k, start_pos=start_pos)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if return_cache else None

        total_len = k.size(2)
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device).view(1, 1, T, 1)
        k_pos = torch.arange(total_len, device=x.device).view(1, 1, 1, total_len)
        attn_mask = q_pos >= k_pos

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        att = att.masked_fill(~attn_mask, float("-inf"))
        att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
        att = self.dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)

        if return_cache:
            return y, new_cache
        return y
