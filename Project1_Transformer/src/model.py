"""M3：TransformerClassifier + load_for_eval 工厂函数。

---- 接口约定（eval/run.py 自检依赖）----
  class TransformerClassifier(nn.Module)
  load_for_eval(ckpt_path: str) -> (model, tokenize_fn)
    其中 tokenize_fn(text: str) -> torch.LongTensor 形状 (T,)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from src.block import TransformerBlock


# ================================================================
#  字符级 Tokenizer（不使用任何预训练模型/分词器）
# ================================================================

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]


class CharTokenizer:
    """字符级 tokenizer：逐字切分 + 特殊 token。

    训练时用 CharTokenizer.build_from_texts() 从语料构建词表。
    保存/加载用 to_dict() / from_dict()。
    """

    def __init__(self, vocab: dict[str, int]):
        # TODO: 保存 vocab (dict: char → id)
        # TODO: 构建反向映射 self.idx_to_char = {id: char}
        # TODO: 保存 self.pad_id, self.unk_id, self.cls_id
        self.vocab = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.pad_id = vocab[PAD_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]
        self.cls_id = vocab[CLS_TOKEN]

    def __len__(self) -> int:
        # TODO: 返回词表大小
        return len(self.vocab)

    def encode(self, text: str, add_cls: bool = True) -> list[int]:
        """将文本编码为 token id 列表。

        Args:
            text: 输入文本
            add_cls: 是否在开头添加 [CLS]

        Returns:
            token id 列表

        步骤：
            1. 如果 add_cls，先加 [CLS] 的 id
            2. 逐字符查 vocab，找不到用 unk_id
        """
        # ---- 你的代码开始 ----
        # TODO: 实现编码逻辑（约 6 行）
        ids = []
        if add_cls:
            ids.append(self.cls_id)
        for char in text:
            ids.append(self.vocab.get(char, self.unk_id))
        return ids
        # ---- 你的代码结束 ----

    def decode(self, ids: list[int]) -> str:
        """将 token id 列表解码回文本。"""
        # TODO: 逐 id 查 self.idx_to_char，拼接返回
        text = []
        for id in ids:
            text.append(self.idx_to_char.get(id, UNK_TOKEN))
        return "".join(text)

    # ---- 工厂方法 ----

    @classmethod
    def build_from_texts(cls, texts: list[str], min_freq: int = 1) -> "CharTokenizer":
        """从文本列表构建词表。

        Args:
            texts: 所有训练文本
            min_freq: 字符最小出现次数

        步骤：
            1. 统计字符频率（Counter）
            2. 先加入 SPECIAL_TOKENS
            3. 按频率从高到低加入满足 min_freq 的字符
            4. 返回 CharTokenizer(vocab)
        """
        # ---- 你的代码开始 ----
        # TODO: 实现词表构建（约 10 行）
        # counter = Counter()
        # for text in texts:
        #     counter.update(text)
        # vocab = {}
        # for tok in SPECIAL_TOKENS:
        #     vocab[tok] = len(vocab)
        # for ch, cnt in counter.most_common():
        #     if cnt >= min_freq and ch not in vocab:
        #         vocab[ch] = len(vocab)
        # return cls(vocab)
        counter = Counter()
        for text in texts:
            counter.update(text)
        vocab = {}
        for tok in SPECIAL_TOKENS:
            vocab[tok] = len(vocab)
        for ch, cnt in counter.most_common():
            if cnt >= min_freq and ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)
        # ---- 你的代码结束 ----

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """序列化为 dict，用于保存到 checkpoint。"""
        # TODO: return {"vocab": self.vocab}
        return {"vocab": dict(self.vocab.items())}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        """从 dict 恢复 tokenizer。"""
        # TODO: return cls(vocab=d["vocab"])
        return cls(vocab=d["vocab"])


# ================================================================
#  位置编码（Positional Encoding）
# ================================================================

class PositionalEncoding(nn.Module):
    """正弦/余弦绝对位置编码。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    编码矩阵注册为 buffer（不参与训练），形状 (1, max_len, d_model)。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
            dropout: 位置编码后的 dropout
        """
        super().__init__()
        # ---- 你的代码开始 ----
        # TODO: 创建位置编码矩阵 pe (max_len, d_model)
        #   1. position = arange(0, max_len).unsqueeze(1)          → (max_len, 1)
        #   2. div_term = exp(arange(0, d_model, 2) * (-log(10000)/d_model))
        #   3. pe[:, 0::2] = sin(position * div_term)   ← 偶数位
        #   4. pe[:, 1::2] = cos(position * div_term)   ← 奇数位
        #   5. register_buffer("pe", pe.unsqueeze(0))              → (1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        # TODO: self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        # ---- 你的代码结束 ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)，加上位置编码后返回。"""
        B, T, _ = x.shape
        return self.dropout(x + self.pe[:, :T, :].to(x.device))


# ================================================================
#  RoPE（Rotary Positional Encoding，S4）
# ================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将 head_dim 后半维取反后与前半维拼接，用于 RoPE 旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 施加 RoPE。形状均为 (B, H, T, d_k)。"""
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE 频率缓存；在 attention 内对 Q/K 旋转，替代绝对 PE。"""

    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.max_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q/k: (B, H, T, d_k)"""
        seq_len = q.size(2)
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        return apply_rotary_pos_emb(q, k, cos, sin)


# ================================================================
#  分类模型
# ================================================================

class TransformerClassifier(nn.Module):
    """Transformer encoder 情感分类器。

    结构：
        TokenEmbedding + PositionalEncoding（或 RoPE 作用于 Q/K）
        → N × TransformerBlock
        → final LayerNorm → [CLS] pooling → Linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        n_classes: int = 2,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_residual: bool = True,
        use_layernorm: bool = True,
        pe_type: str = "abs",
    ):
        """
        Args:
            vocab_size: 词表大小
            d_model:    模型维度
            n_heads:    注意力头数
            n_layers:   Transformer 层数
            d_ff:       FFN 中间层维度
            n_classes:  分类类别数（二分类=2）
            max_len:    最大序列长度
            dropout:    dropout 概率
            pad_idx:    PAD token 的 id
            pe_type:    位置编码类型，"abs"（绝对 PE）或 "rope"
        """
        super().__init__()
        if pe_type not in ("abs", "rope"):
            raise ValueError(f"pe_type must be 'abs' or 'rope', got {pe_type!r}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.pe_type = pe_type
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if pe_type == "abs":
            self.position_encoding = PositionalEncoding(d_model, max_len, dropout)
            self.rotary_emb = None
        else:
            self.position_encoding = nn.Identity()
            self.embed_dropout = nn.Dropout(dropout)
            self.rotary_emb = RotaryEmbedding(d_model // n_heads, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_residual=use_residual,
                use_layernorm=use_layernorm,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self._config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "max_len": max_len,
            "pad_idx": pad_idx,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "n_classes": n_classes,
            "dropout": dropout,
            "use_residual": use_residual,
            "use_layernorm": use_layernorm,
            "pe_type": pe_type,
        }

    def _create_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """构建 padding mask：True = PAD 位置需屏蔽。

        Returns:
            (B, 1, 1, T) —— 广播到 (B, H, T, T)
        """
        # TODO: mask = (input_ids == self.pad_idx)   → (B, T)
        mask = (input_ids == self.pad_idx)
        # TODO: return mask.unsqueeze(1).unsqueeze(2) → (B, 1, 1, T)
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: (B, T) token id 序列
            mask:      可选手动 attention mask（用于 causal mask 等场景）

        Returns:
            logits: (B, n_classes)

        步骤：
            1. 无 mask 时自动生成 padding mask
            2. token embedding（乘以 sqrt(d_model) 缩放）
            3. 位置编码
            4. 逐层 TransformerBlock
            5. final LayerNorm → 取 [CLS] 位 → Linear 分类
        """
        # ---- 你的代码开始 ----
        # TODO: 实现上述 5 步
        # if mask is None:
        #     mask = self._create_padding_mask(input_ids)
        # x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # x = self.position_encoding(x)
        # for block in self.blocks:
        #     x = block(x, mask)
        # x = self.final_norm(x)
        # cls_repr = x[:, 0, :]                    # [CLS] 在序列首位
        # return self.classifier(cls_repr)
        if mask is None:
            mask = self._create_padding_mask(input_ids)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        if self.pe_type == "abs":
            x = self.position_encoding(x)
        else:
            x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x, mask, rotary_emb=self.rotary_emb)
        x = self.final_norm(x)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)


# ======================================``==========================
#  factory：eval/run.py 自检入口
# ================================================================

def load_for_eval(ckpt_path: str):
    """从 checkpoint 加载模型和 tokenize 函数。

    Args:
        ckpt_path: best.pt 的路径

    Returns:
        (model, tokenize_fn)

    其中 tokenize_fn(text: str) -> torch.LongTensor 形状 (T,)
    """
    # ---- 你的代码开始 ----
    # TODO: 1. torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # TODO: 2. 从 ckpt["tokenizer"] 恢复 CharTokenizer
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    # TODO: 3. 从 ckpt["config"] 取出配置，重建 TransformerClassifier
    config = dict(ckpt["config"])
    config.setdefault("pe_type", "abs")
    model = TransformerClassifier(**config)
    # TODO: 4. model.load_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    # TODO: 5. 定义 tokenize_fn:
    #         def tokenize_fn(text):
    #             ids = tokenizer.encode(text, add_cls=True)
    #             ids = ids[:max_len]
    #             return torch.tensor(ids, dtype=torch.long)
    def tokenize_fn(text):
        ids = tokenizer.encode(text, add_cls=True)
        ids = ids[:config["max_len"]]
        return torch.tensor(ids, dtype=torch.long)
    # TODO: 6. return model, tokenize_fn
    return model, tokenize_fn
    # ---- 你的代码结束 ----


# ================================================================
#  辅助：保存 checkpoint（train.py 会调用）
# ================================================================

def save_checkpoint(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    ckpt_dir: str | Path,
    filename: str = "best.pt",
    **extra,
):
    """保存完整 checkpoint（model_state_dict + config + tokenizer + 额外信息）。

    Args:
        model:     TransformerClassifier 实例
        tokenizer: CharTokenizer 实例
        ckpt_dir:  保存目录（不存在会自动创建）
        filename:  文件名
        **extra:   额外要保存的字段（如 epoch, dev_acc 等）
    """
    # ---- 你的代码开始 ----
    # TODO: 1. Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # TODO: 2. 组装 payload:
    #         {"model_state_dict": model.state_dict(),
    #          "config": model._config,
    #          "tokenizer": tokenizer.to_dict(),
    #          **extra}
    # TODO: 3. torch.save(payload, ckpt_dir / filename)
    # TODO: 4. print 保存路径
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "config": model._config, "tokenizer": tokenizer.to_dict(), **extra}
    torch.save(payload, ckpt_dir / filename)
    print(f"Checkpoint saved to {ckpt_dir / filename}")
    # ---- 你的代码结束 ----
