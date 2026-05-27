"""任务四：重排序器（bge-reranker 精排）。"""

from typing import List, Dict


class Reranker:
    """使用 bge-reranker 对检索结果精排。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """初始化。

        Args:
            model_name: reranker 模型名称
        """
        # TODO: 加载 CrossEncoder（如 FlagEmbedding 或 sentence-transformers 的 CrossEncoder）
        raise NotImplementedError("TODO: 实现 Reranker.__init__")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """对检索结果重排序。

        Args:
            query: 查询文本
            results: 初检结果列表（来自 Retriever.retrieve）
            top_k: 返回的精排结果数

        Returns:
            List[Dict]: 精排后的结果（text, score, source），按分数降序
        """
        # TODO: 1. 对每个 (query, chunk) 对计算相关性分数
        # TODO: 2. 按分数降序排序
        # TODO: 3. 返回 top_k 结果
        raise NotImplementedError("TODO: 实现 rerank")
