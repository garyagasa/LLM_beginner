"""任务四：检索器。

接口约定（eval/run.py 会自动检测）：
  - class Retriever 含 retrieve(query: str, k: int) -> List[dict]
    每个 dict 含 text, score, source
"""

from typing import List, Dict


class Retriever:
    """基于 embedding + FAISS 的检索器。"""

    def __init__(self, indexer, embedding_model):
        """初始化。

        Args:
            indexer: Indexer 实例（已构建或加载的索引）
            embedding_model: sentence-transformers 模型
        """
        # TODO: 保存 indexer 和 embedding_model
        raise NotImplementedError("TODO: 实现 Retriever.__init__")

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """检索与 query 最相关的 k 个 chunk。

        Args:
            query: 查询文本
            k: 返回的 chunk 数量

        Returns:
            List[Dict]: 每个元素包含:
                - text: str   chunk 文本
                - score: float  相似度分数
                - source: str  来源文档名
        """
        # TODO: 1. 对 query 计算 embedding
        # TODO: 2. 在 FAISS 索引中搜索 top-k
        # TODO: 3. 返回对应的 chunk 文本和分数
        raise NotImplementedError("TODO: 实现 retrieve")
