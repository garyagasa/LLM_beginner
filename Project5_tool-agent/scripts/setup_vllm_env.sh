#!/usr/bin/env bash
# 创建独立的 vLLM conda 环境（与写 Agent 的 llm_pj5 分开）
#
# 本机：Driver 560 + CUDA 12.6 → 必须用 PyTorch cu126，不能用 pip 默认的 cu130。
#
# 用法：
#   bash scripts/setup_vllm_env.sh
#   conda activate vllm_pj5
#   bash scripts/serve_vllm.sh
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm_pj5}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "    env ${ENV_NAME} already exists — will install/upgrade packages inside it"
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch (CUDA 12.6 wheels)"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url "${TORCH_CUDA_INDEX}"

echo "==> Installing vLLM stack (pinned transformers 4.51.x — 勿用 5.x)"
pip install -r "${ROOT}/requirements-vllm.txt"
# 防止后续依赖把 transformers 又升到 5
pip install "transformers==4.51.3" "huggingface-hub>=0.30,<1.0"

echo "==> Sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_compiled", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — check driver / GPU")
if torch.version.cuda and not str(torch.version.cuda).startswith("12."):
    print("WARNING: expected CUDA 12.x build, got", torch.version.cuda)
print("gpu0", torch.cuda.get_device_name(0))
import vllm
print("vllm", vllm.__version__)
print("OK")
PY

echo
echo "Done. Next:"
echo "  conda activate ${ENV_NAME}"
echo "  export HF_ENDPOINT=https://hf-mirror.com   # optional"
echo "  bash scripts/serve_vllm.sh"
echo
echo "Agent 代码仍在 llm_pj5 里跑："
echo "  conda activate llm_pj5"
echo "  pip install -r requirements.txt"
echo "  python eval/run.py"
