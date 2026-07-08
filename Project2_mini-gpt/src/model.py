"""Decoder-only mini-GPT with RoPE and KV cache."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from src.attention import CausalSelfAttention
from src.sampling import sample_next_token
from src.tokenizer import BPETokenizer


@dataclass
class MiniGPTConfig:
    vocab_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    dropout: float = 0.1


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            config.n_embd, config.n_head, config.block_size, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        start_pos: int = 0,
        return_cache: bool = False,
    ):
        if return_cache:
            attn_out, new_cache = self.attn(
                self.ln1(x), kv_cache=kv_cache, start_pos=start_pos, return_cache=True
            )
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, new_cache
        attn_out = self.attn(
            self.ln1(x), kv_cache=kv_cache, start_pos=start_pos, return_cache=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig | dict):
        super().__init__()
        if isinstance(config, dict):
            config = MiniGPTConfig(**config)
        self.config = config
        self.block_size = config.block_size
        self.max_seq_len = config.block_size

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ids: torch.Tensor,
        kv_cache: List | None = None,
        return_cache: bool = False,
    ):
        B, T = ids.shape
        if kv_cache is None:
            start_pos = 0
            layer_caches = [None] * len(self.blocks)
        else:
            start_pos = kv_cache[0][0].size(2) if kv_cache[0] is not None else 0
            layer_caches = kv_cache

        x = self.drop(self.tok_emb(ids))
        new_caches = []
        for i, block in enumerate(self.blocks):
            if return_cache:
                x, lc = block(x, layer_caches[i], start_pos, return_cache=True)
                new_caches.append(lc)
            else:
                x = block(x, layer_caches[i], start_pos, return_cache=False)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        top_p: float | None = 0.9,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        ids = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids.clone()
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                logits, cache = self.forward(ids, return_cache=True)
            else:
                logits, cache = self.forward(ids[:, -1:], kv_cache=cache, return_cache=True)
            next_id = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            ids = torch.cat([ids, next_id], dim=1)
            if ids.size(1) >= self.block_size:
                break
        return ids[0]


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    tokenizer_path: str | Path,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "tokenizer_path": str(tokenizer_path),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_for_eval(ckpt_path: str | Path) -> tuple[MiniGPT, BPETokenizer]:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MiniGPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    tok_path = ckpt.get("tokenizer_path", str(ckpt_path.parent / "tokenizer.json"))
    tokenizer = BPETokenizer.from_pretrained(tok_path)
    model.eval()
    return model, tokenizer
