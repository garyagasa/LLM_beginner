"""模型与运行配置（一般无需修改）。

默认对接 vLLM 的 OpenAI 兼容接口：
  python -m vllm.entrypoints.openai.api_server ...  →  http://localhost:8000/v1
可用环境变量覆盖：
  OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL / REACT_MAX_STEPS

也可用 Ollama（改 BASE_URL=http://localhost:11434/v1, MODEL=qwen2.5:7b-instruct）。
"""

from __future__ import annotations

import os


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "10"))
TEMPERATURE = float(os.getenv("REACT_TEMPERATURE", "0.1"))
