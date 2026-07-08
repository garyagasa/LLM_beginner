"""M4：DPO 训练 —— 在 SFT LoRA 基础上做偏好对齐。

用法：
    python train_sft.py                    # 先完成 SFT，产出 ckpt/sft/
    # 准备 DPO 偏好数据（见 README / data/download.py 提示）
    python train_dpo.py
    python train_dpo.py --sft-ckpt ckpt/sft --beta 0.1

输出：
    ckpt/dpo/          DPO 后的 LoRA 权重

要点：
    - reference model = SFT 后的策略快照，全程 freeze，只 forward
    - policy model 从 ckpt/sft 加载 LoRA 并继续训练
    - 每条样本 4 次 forward：policy(chosen/rejected) + ref(chosen/rejected)
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, load_lora_state_dict, lora_state_dict
from src.dataset import DPODataset
from src.dpo import dpo_loss, _sequence_log_probs

DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"
DEFAULT_SFT_CKPT = ROOT / "ckpt" / "sft"
DEFAULT_DPO_CKPT = ROOT / "ckpt" / "dpo"


def parse_args():
    p = argparse.ArgumentParser(description="DPO after SFT (hand-written LoRA)")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--data", type=str, required=False,
                   help="DPO jsonl 路径，如 data/dpo/train.jsonl")
    p.add_argument("--sft-ckpt", type=str, default=str(DEFAULT_SFT_CKPT))
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_DPO_CKPT))
    p.add_argument("--beta", type=float, default=0.1, help="DPO 温度 β")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="llm-beginner-p3-dpo")
    return p.parse_args()


def build_ref_model(policy_model: torch.nn.Module) -> torch.nn.Module:
    """深拷贝 policy 作为 reference model 并 freeze。"""
    # TODO: ref = copy.deepcopy(policy_model); ref.eval(); 所有参数 requires_grad=False
    raise NotImplementedError("TODO: 实现 build_ref_model")


def train_one_epoch(policy, ref, loader, optimizer, device, beta: float) -> float:
    """DPO 训练循环。"""
    policy.train()
    ref.eval()

    # TODO: 对 batch 中 chosen/rejected 分别 forward policy 与 ref
    # TODO: 用 _sequence_log_probs 得到 logps，再 dpo_loss(...)
    # TODO: 只更新 policy 的 LoRA 参数
    raise NotImplementedError("TODO: 实现 DPO train_one_epoch")

    return 0.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not Path(args.model).exists():
        sys.exit(f"[错误] 基座模型不存在: {args.model}")
    if not Path(args.sft_ckpt).exists():
        sys.exit(f"[错误] SFT checkpoint 不存在: {args.sft_ckpt}\n请先运行: python train_sft.py")

    # TODO:
    # 1. 加载 base model + tokenizer
    # 2. inject_lora + load_lora_state_dict(sft)
    # 3. policy = model；ref = build_ref_model(policy)
    # 4. DPODataset + DataLoader
    # 5. 训练并保存到 ckpt/dpo/

    print("TODO: 完成 train_dpo.py 后可用 src/compare.py 对比 base / SFT / DPO 输出")


if __name__ == "__main__":
    main()
