"""任务四：索引构建（embedding + FAISS 索引）。"""

from typing import List
import numpy as np
import faiss


class Indexer:
    """将 chunk 列表构建为 FAISS 向量索引。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """初始化。

        Args:
            model_name: embedding 模型名称（默认 BGE 中文小模型）
        """
        # TODO: 1. 加载 sentence-transformers 模型
        # TODO: 2. 初始化空的 FAISS 索引
        raise NotImplementedError("TODO: 实现 Indexer.__init__")

    def build(self, chunks: List[str], batch_size: int = 32):
        """为 chunk 列表构建索引。

        Args:
            chunks: 文本块列表
            batch_size: embedding 批次大小
        """
        # TODO: 1. 批量计算 embedding
        # TODO: 2. 构建 FAISS 索引（如 IndexFlatIP 用于内积搜索）
        # TODO: 3. 可选：做 L2 归一化后用内积 = 余弦相似度
        raise NotImplementedError("TODO: 实现 build")

    def save(self, path: str):
        """保存索引到文件。

        Args:
            path: 保存路径
        """
        # TODO: 1. 保存 FAISS 索引（faiss.write_index）
        # TODO: 2. 保存 chunk 列表（JSON）
        raise NotImplementedError("TODO: 实现 save")

    def load(self, path: str):
        """从文件加载索引。

        Args:
            path: 索引路径
        """
        # TODO: 1. 读取 FAISS 索引
        # TODO: 2. 读取 chunk 列表
        raise NotImplementedError("TODO: 实现 load")
