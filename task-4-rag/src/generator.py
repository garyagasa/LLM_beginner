"""任务四：基于检索结果 + LLM 生成答案。"""

from typing import List, Dict
from openai import OpenAI


class Generator:
    """将检索结果拼接为 prompt，调用本地 LLM 生成答案。"""

    def __init__(self, api_base: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b-instruct"):
        """初始化。

        Args:
            api_base: OpenAI 兼容 API 地址（默认 Ollama 本地部署）
            model: 模型名称
        """
        # TODO: 1. 初始化 OpenAI 客户端（指向本地 API）
        # TODO: 2. 保存模型名称
        raise NotImplementedError("TODO: 实现 Generator.__init__")

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """基于检索上下文生成答案。

        Args:
            query: 用户问题
            contexts: 检索/精排后的上下文列表

        Returns:
            str: 生成的答案
        """
        # TODO: 1. 拼接 prompt 模板：
        #     "请根据以下参考资料回答问题。\n\n参考资料：\n{contexts}\n\n问题：{query}\n\n回答："
        # TODO: 2. 调用 LLM 生成
        # TODO: 3. 返回生成的文本
        raise NotImplementedError("TODO: 实现 generate")
