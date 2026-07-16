#!/usr/bin/env bash
# 启动 vLLM OpenAI 兼容服务（默认 http://localhost:8000/v1）
#
# 请在 vllm_pj5 环境中运行（不要用装了 cu130 torch 的 llm_pj5）：
#   conda activate vllm_pj5
#   bash scripts/serve_vllm.sh
#
# 其它：
#   CUDA_VISIBLE_DEVICES=1 bash scripts/serve_vllm.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/serve_vllm.sh
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 启动前快速检查：避免再次踩 cu130 + 旧驱动
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("ERROR: 当前 python 没有 torch。请先: conda activate vllm_pj5", file=sys.stderr)
    sys.exit(1)
print(f"torch={torch.__version__} cuda_compiled={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA 不可用。若报 driver too old，说明装成了 cu130；请重跑 scripts/setup_vllm_env.sh", file=sys.stderr)
    sys.exit(1)
ver = (torch.__version__ or "") + " " + (torch.version.cuda or "")
if "cu130" in ver or (torch.version.cuda or "").startswith("13"):
    print("ERROR: 检测到 CUDA 13 / cu130 构建，与本机驱动 12.6 不兼容。", file=sys.stderr)
    print("       请: conda activate vllm_pj5 && bash scripts/setup_vllm_env.sh", file=sys.stderr)
    sys.exit(1)
try:
    import vllm  # noqa: F401
except ImportError:
    print("ERROR: 未安装 vllm。请: bash scripts/setup_vllm_env.sh", file=sys.stderr)
    sys.exit(1)
PY

echo "Starting vLLM: model=${MODEL} port=${PORT} GPU=${CUDA_VISIBLE_DEVICES}"
echo "Client (llm_pj5) should use:"
echo "  export OPENAI_BASE_URL=http://localhost:${PORT}/v1"
echo "  export OPENAI_API_KEY=EMPTY"
echo "  export OPENAI_MODEL=${MODEL}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  "$@"
