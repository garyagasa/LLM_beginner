"""任务四：端到端 RAG pipeline。

接口约定（eval/run.py 会自动检测）：
  - answer(query: str) -> dict(answer: str, sources: List[dict])
"""

from typing import List, Dict


def answer(query: str) -> Dict:
    """端到端 RAG：接收问题，返回答案 + 引用来源。

    Args:
        query: 用户问题

    Returns:
        dict: {
            "answer": str,     生成答案
            "sources": [       引用的来源
                {"text": str, "score": float, "source": str},
                ...
            ]
        }
    """
    # TODO: 1. 加载/初始化 Indexer, Retriever, Reranker, Generator
    # TODO: 2. 检索 → 精排（可选）→ 生成
    # TODO: 3. 返回答案和来源
    raise NotImplementedError("TODO: 实现 answer")
