"""M4：DPO 训练 —— 在 SFT LoRA 基础上做偏好对齐。"""

from __future__ import annotations

import argparse
import copy
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.lora import inject_lora, load_lora_state_dict, lora_state_dict
from src.dataset import DPODataset, dpo_pad_collate
from src.dpo import dpo_loss, _sequence_log_probs
from src.wandb_utils import init_wandb, wandb_log, wandb_finish

DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"
DEFAULT_SFT_CKPT = ROOT / "ckpt" / "sft"
DEFAULT_DPO_CKPT = ROOT / "ckpt" / "dpo"
DEFAULT_DPO_DATA = ROOT / "data" / "dpo" / "train.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="DPO after SFT (hand-written LoRA)")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--data", type=str, default=str(DEFAULT_DPO_DATA),
                   help="DPO jsonl 路径，如 data/dpo/train.jsonl")
    p.add_argument("--sft-ckpt", type=str, default=str(DEFAULT_SFT_CKPT))
    p.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_DPO_CKPT))
    p.add_argument("--beta", type=float, default=0.1, help="DPO 温度 β")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10, help="每 N step 记录一次 wandb")
    p.add_argument("--no-wandb", action="store_true", help="禁用 Weights & Biases 日志")
    p.add_argument("--wandb-project", type=str, default="llm-beginner-p3-dpo")
    p.add_argument("--wandb-run-name", type=str, default=None, help="run 名称，默认自动生成")
    p.add_argument("--wandb-entity", type=str, default=None, help="wandb 团队/用户名，默认用已登录账号")
    return p.parse_args()


def build_ref_model(policy_model: torch.nn.Module) -> torch.nn.Module:
    """深拷贝 policy 作为 reference model 并 freeze。"""
    ref = copy.deepcopy(policy_model)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad = False
    return ref


def _forward_log_probs(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _sequence_log_probs(outputs.logits, labels, attention_mask)


def train_one_epoch(
    policy,
    ref,
    loader,
    optimizer,
    device,
    beta: float,
    *,
    epoch: int = 1,
    log_every: int = 10,
    global_step: int = 0,
) -> tuple[float, int]:
    """DPO 训练循环。返回 (avg_loss, 更新后的 global_step)。"""
    policy.train()
    ref.eval()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"DPO epoch {epoch}")
    for batch in progress:
        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)

        policy_chosen_logps = _forward_log_probs(
            policy, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = _forward_log_probs(
            policy, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        with torch.no_grad():
            ref_chosen_logps = _forward_log_probs(
                ref, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = _forward_log_probs(
                ref, rejected_input_ids, rejected_attention_mask, rejected_labels
            )

        loss, metrics = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            margin=f"{metrics['reward_margin']:.4f}",
        )

        if global_step % log_every == 0:
            wandb_log(
                {
                    "train/dpo_loss": loss.item(),
                    "train/reward_margin": metrics["reward_margin"],
                    "train/accuracy": metrics["accuracy"],
                    "train/chosen_logps": policy_chosen_logps.mean().item(),
                    "train/rejected_logps": policy_rejected_logps.mean().item(),
                    "train/epoch": epoch,
                },
                step=global_step,
            )

    return total_loss / max(len(loader), 1), global_step


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not Path(args.model).exists():
        sys.exit(f"[错误] 基座模型不存在: {args.model}")
    if not Path(args.sft_ckpt).exists():
        sys.exit(f"[错误] SFT checkpoint 不存在: {args.sft_ckpt}\n请先运行: python train_sft.py")

    lora_path = Path(args.sft_ckpt) / "lora.pt"
    if not lora_path.exists():
        sys.exit(f"[错误] 未找到 SFT LoRA 权重: {lora_path}")

    init_wandb(args, extra={"beta": args.beta})

    dtype = torch.bfloat16 if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    ) else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.gradient_checkpointing_enable()

    target = [m.strip() for m in args.target_modules.split(",")]
    inject_lora(model, target_modules=target, r=args.lora_r, alpha=args.lora_alpha)
    load_lora_state_dict(model, torch.load(lora_path, map_location=args.device))

    policy = model
    ref = build_ref_model(policy)

    dataset = DPODataset(
        args.data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    collate = partial(dpo_pad_collate, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
    )

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    avg_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, global_step = train_one_epoch(
            policy,
            ref,
            loader,
            optimizer,
            args.device,
            args.beta,
            epoch=epoch,
            log_every=args.log_every,
            global_step=global_step,
        )
        wandb_log({"train/epoch_dpo_loss": avg_loss, "train/epoch": epoch}, step=global_step)
        best_loss = min(best_loss, avg_loss)
        print(f"[DPO] epoch {epoch} avg_loss={avg_loss:.4f}")

    ckpt_path = Path(args.ckpt_dir) / "lora.pt"
    torch.save(lora_state_dict(policy), ckpt_path)
    print(f"[DPO] 已保存 LoRA 权重到 {ckpt_path}")

    wandb_finish(
        summary={
            "final_dpo_loss": avg_loss,
            "best_dpo_loss": best_loss,
            "total_steps": global_step,
        }
    )


if __name__ == "__main__":
    main()
