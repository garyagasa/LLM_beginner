"""任务二：decoder-only Mini-GPT 完整模型。

接口约定（eval/run.py 会自动检测）：
  - class MiniGPT(nn.Module) 含 forward, generate
  - load_for_eval(ckpt_path) -> (model, tokenizer)
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CausalMultiHeadAttention


class DecoderBlock(nn.Module):
    """一个 Transformer decoder block：attention + FFN + residual + LayerNorm。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        # TODO: 初始化 self.attention, self.norm1, self.norm2, FFN, dropout
        super().__init__()
        raise NotImplementedError("TODO: 实现 DecoderBlock.__init__")

    def forward(self, x, kv_cache=None, return_cache=False):
        # TODO: 同 task-1 TransformerBlock 但用的是 causal attention + KV cache
        raise NotImplementedError("TODO: 实现 DecoderBlock.forward")


class MiniGPT(nn.Module):
    """decoder-only mini-GPT，支持自回归生成。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 2048,
        dropout: float = 0.1,
    ):
        """初始化。

        Args:
            vocab_size: 词表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: decoder block 层数
            d_ff: FFN 中间层维度
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        # TODO: 1. self.token_embedding = nn.Embedding(vocab_size, d_model)
        # TODO: 2. self.blocks = nn.ModuleList([DecoderBlock(...) for _ in range(n_layers)])
        # TODO: 3. self.ln_f = nn.LayerNorm(d_model)
        # TODO: 4. self.lm_head = nn.Linear(d_model, vocab_size)（通常与 embedding 共享权重）
        super().__init__()
        raise NotImplementedError("TODO: 实现 MiniGPT.__init__")

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: list | None = None,
        return_cache: bool = False,
    ):
        """前向传播。

        Args:
            input_ids: (B, T)  token ids
            kv_cache: 可选的每层 KV cache 列表
            return_cache: 是否返回更新后的 KV cache

        Returns:
            若 return_cache=False: logits (B, T, vocab_size)
            若 return_cache=True:  (logits, new_kv_cache)
        """
        # TODO: 1. token embedding
        # TODO: 2. 逐层过 DecoderBlock
        # TODO: 3. ln_f → lm_head
        raise NotImplementedError("TODO: 实现 MiniGPT.forward")

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 100,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> List[int]:
        """自回归生成。

        Args:
            prompt_ids: 提示 token id 列表
            max_new_tokens: 最大新生成 token 数
            top_p: nucleus sampling 阈值
            temperature: 温度系数

        Returns:
            List[int]: 完整 token id 序列（prompt + 生成）
        """
        # TODO: 1. 将 prompt_ids 转为 tensor (1, T)
        # TODO: 2. 使用 KV cache 逐步生成
        # TODO: 3. 每步对 logits 做 temperature scaling + top-p 采样
        # TODO: 4. 遇到 EOS token 或达到 max_new_tokens 停止
        raise NotImplementedError("TODO: 实现 generate")


def load_for_eval(ckpt_path: str):
    """加载模型用于评测。

    Args:
        ckpt_path: checkpoint 路径

    Returns:
        model: MiniGPT（已加载权重，eval 模式）
        tokenizer: BPETokenizer 实例
    """
    # TODO: 1. 加载 tokenizer.json
    # TODO: 2. 初始化 MiniGPT（参数需与训练时一致）
    # TODO: 3. 加载 state_dict
    # TODO: 4. model.eval()
    # TODO: 5. return model, tokenizer
    raise NotImplementedError("TODO: 实现 load_for_eval")
