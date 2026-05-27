"""任务二：BPE tokenizer。

接口约定（eval/run.py 会自动检测）：
  - class BPETokenizer 含 encode(text) -> List[int], decode(ids) -> str,
    vocab_size, from_pretrained(path)
"""

from typing import List
import json


class BPETokenizer:
    """BPE tokenizer，手写 merge 过程。"""

    def __init__(self):
        # TODO: 初始化 merges dict、vocab dict
        raise NotImplementedError("TODO: 实现 BPETokenizer.__init__")

    @property
    def vocab_size(self) -> int:
        """返回词表大小。"""
        # TODO: 返回 len(self.vocab)
        raise NotImplementedError("TODO: 实现 vocab_size")

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 列表。

        Args:
            text: 输入文本

        Returns:
            List[int]: token id 列表
        """
        # TODO: 1. 将 text 转为字节序列
        # TODO: 2. 按规则字节对合并（反复应用 merges 到不能再合并）
        # TODO: 3. 映射为 id 列表
        raise NotImplementedError("TODO: 实现 encode")

    def decode(self, ids: List[int]) -> str:
        """将 token id 列表解码为文本。

        Args:
            ids: token id 列表

        Returns:
            str: 解码后文本
        """
        # TODO: 1. id → token byte 序列
        # TODO: 2. 拼接并 UTF-8 解码
        raise NotImplementedError("TODO: 实现 decode")

    def train(self, texts: List[str], vocab_size: int):
        """在给定文本上训练 BPE。

        Args:
            texts: 训练文本列表
            vocab_size: 目标词表大小
        """
        # TODO: 1. 初始化 byte-level vocab（256 字节 + 1 个 <|endoftext|>）
        # TODO: 2. 将文本转为 byte 序列
        # TODO: 3. 反复统计 pair 频率 → 合并最高频 pair → 加入 merges 和 vocab
        # TODO: 4. 直到 vocab 达到目标大小
        raise NotImplementedError("TODO: 实现 train")

    def save(self, path: str):
        """保存词表到文件。"""
        # TODO: 保存 merges + vocab 为 JSON
        raise NotImplementedError("TODO: 实现 save")

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        """从文件加载词表。

        Args:
            path: 词表文件路径

        Returns:
            BPETokenizer 实例
        """
        # TODO: 1. 读取 JSON 文件
        # TODO: 2. 恢复 merges 和 vocab
        # TODO: 3. 返回 tokenizer 实例
        raise NotImplementedError("TODO: 实现 from_pretrained")
