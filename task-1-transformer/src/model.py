"""任务一：Transformer 分类器 + 加载工厂函数。

接口约定（eval/run.py 会自动检测）：
  - class TransformerClassifier(nn.Module)
  - load_for_eval(ckpt_path: str) -> (model, tokenize_fn)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from .block import TransformerBlock


class TransformerClassifier(nn.Module):
    """堆叠 N 个 TransformerBlock + pooling + 分类头。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        n_classes: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """初始化。

        Args:
            vocab_size: 词表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer block 层数
            d_ff: FFN 中间层维度
            n_classes: 分类类别数（ChnSentiCorp 二分类 = 2）
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        # TODO: 1. self.embedding = nn.Embedding(vocab_size, d_model)
        # TODO: 2. self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        # TODO: 3. self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        # TODO: 4. self.norm = nn.LayerNorm(d_model)
        # TODO: 5. self.classifier = nn.Linear(d_model, n_classes)
        # TODO: 6. self.dropout = nn.Dropout(dropout)
        super().__init__()
        raise NotImplementedError("TODO: 实现 TransformerClassifier.__init__")

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None):
        """前向传播。

        Args:
            input_ids: (B, T)  token ids
            mask: attention mask

        Returns:
            logits: (B, n_classes)
        """
        # TODO: 1. x = self.embedding(input_ids) + self.pos_embedding[:, :T, :]
        # TODO: 2. 过每一层 TransformerBlock
        # TODO: 3. pool = x.mean(dim=1)  # mean pooling
        # TODO: 4. logits = self.classifier(self.dropout(pool))
        raise NotImplementedError("TODO: 实现 TransformerClassifier.forward")


def load_for_eval(ckpt_path: str):
    """加载模型用于评测。

    Args:
        ckpt_path: checkpoint 路径

    Returns:
        model: TransformerClassifier（已加载权重，eval 模式）
        tokenize_fn: 接受 str 返回 token ids 的函数
    """
    # TODO: 1. 加载 tokenizer（用 transformers 的 AutoTokenizer，如 bert-base-chinese）
    # TODO: 2. 初始化 TransformerClassifier（vocab_size 需与 tokenizer 一致）
    # TODO: 3. 加载 state_dict
    # TODO: 4. model.eval()
    # TODO: 5. 定义 tokenize_fn = lambda text: tokenizer.encode(text, ...)
    # TODO: 6. return model, tokenize_fn
    raise NotImplementedError("TODO: 实现 load_for_eval")
