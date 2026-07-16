"""LLM 客户端薄封装：对接 OpenAI 兼容 API（Ollama / vLLM / llama.cpp）。

一般只需调用 chat()；模型地址见 src/config.py。
openai 仅在真正调模型时导入，方便先单独完成 M1 工具。
"""

from __future__ import annotations

from typing import Any

from . import config


def get_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "缺少 openai 包，请先: pip install -r requirements.txt"
        ) from e
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    stop: list[str] | None = None,
) -> str:
    """发送 chat completion，返回 assistant 文本。

    TODO(M2，可选)：如需自定义 stop / max_tokens，在这里扩展参数即可。
    默认实现已可用，可不改。
    """
    client = get_client()
    resp = client.chat.completions.create(
        model=model or config.MODEL,
        messages=messages,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        stop=stop,
    )
    return (resp.choices[0].message.content or "").strip()
