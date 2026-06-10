"""训练脚本：在 ChnSentiCorp 上训练 TransformerClassifier。

用法：
    python train.py                # 默认超参
    python train.py --epochs 10    # 覆盖参数

输出：
    ckpt/best.pt         最佳模型
    ckpt/last.pt         最后模型
    figures/             注意力热图（训练完后自动生成 3 张）
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.model import TransformerClassifier, CharTokenizer, save_checkpoint

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpt"
FIG_DIR = ROOT / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="Train TransformerClassifier on ChnSentiCorp")
    # 模型
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=200, help="最大序列长度（字符数）")
    # 训练
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_early_stop", action="store_true", help="禁用 early stopping")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer, max_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids = self.tokenizer.encode(row["text"], add_cls=True)
        ids = ids[: self.max_len]
        label = int(row["label"])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Padding + 组装 batch。"""
    ids_list, labels = zip(*batch)
    max_len = max(ids.shape[0] for ids in ids_list)

    padded = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : ids.shape[0]] = ids

    return padded, torch.stack(labels)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """在 dev set 上计算准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 注意力热图
# ---------------------------------------------------------------------------


def plot_attention_heatmaps(model, tokenizer, sentences, device, save_dir):
    """生成注意力热图并保存到 save_dir。"""
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    for idx, (text, label_name) in enumerate(sentences):
        ids = tokenizer.encode(text, add_cls=True)
        tokens = [tokenizer.idx_to_char.get(i, "[?]") for i in ids]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, T)

        # 需要获取 attention weights —— 这里暂时用简单的方式：
        # 重新跑一遍，手动计算最后一层某个 head 的 attention
        with torch.no_grad():
            B, T = input_ids.shape
            mask = input_ids == tokenizer.pad_id
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

            # Embedding
            x = model.token_embedding(input_ids) * math.sqrt(model.d_model)
            x = model.position_encoding(x)

            # 逐层到最后一层 attention 前
            for i, block in enumerate(model.blocks):
                # 手动计算本层 attention 的权重
                normed = block.norm1(x)
                B = normed.size(0)
                Q = (
                    block.attention.W_q(normed)
                    .view(B, -1, block.attention.n_heads, block.attention.d_k)
                    .transpose(1, 2)
                )
                K = (
                    block.attention.W_k(normed)
                    .view(B, -1, block.attention.n_heads, block.attention.d_k)
                    .transpose(1, 2)
                )

                d_k = Q.size(-1)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))

                # 取最后一个 head 的 attention（或其他 head）
                if i == len(model.blocks) - 1:
                    # 取所有 head 的平均 attention 画图
                    attn_weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
                    avg_attn = attn_weights[0].mean(dim=0).cpu().numpy()  # (T, T)

                    fig, ax = plt.subplots(figsize=(max(8, T * 0.3), max(6, T * 0.25)))
                    im = ax.imshow(avg_attn, cmap="Reds", aspect="auto")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                    # 标签（只标前 30 个 token 避免太密）
                    show_n = min(T, 30)
                    ax.set_xticks(range(show_n))
                    ax.set_xticklabels(tokens[:show_n], rotation=90, fontsize=8)
                    ax.set_yticks(range(show_n))
                    ax.set_yticklabels(tokens[:show_n], fontsize=8)
                    ax.set_xlabel("Key")
                    ax.set_ylabel("Query")
                    ax.set_title(
                        f"Attention Heatmap — Layer {len(model.blocks)}, all-head avg\n"
                        f"Label: {label_name}  |  Text: {text[:40]}..."
                    )
                    fig.tight_layout()
                    fig.savefig(
                        save_dir / f"heatmap_{idx + 1}_{label_name}.png", dpi=150
                    )
                    plt.close(fig)
                    print(f"  [热图] 已保存 {save_dir / f'heatmap_{idx + 1}_{label_name}.png'}")

                # 继续前向
                x = block(x, mask)

    model.train()


# ---------------------------------------------------------------------------
# 主训练
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"设备: {device}")
    print(f"参数: {vars(args)}")

    # ---- 加载数据 ----
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "validation.parquet"
    if not train_path.exists() or not dev_path.exists():
        sys.exit(
            f"数据文件缺失，请先运行: python data/download.py\n"
            f"  期望: {train_path}, {dev_path}"
        )

    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(dev_df)} 条")

    # ---- 构建 tokenizer ----
    print("构建字符级 tokenizer ...")
    tokenizer = CharTokenizer.build_from_texts(train_df["text"].tolist(), min_freq=1)
    print(f"词表大小: {len(tokenizer)}")

    # ---- DataLoader ----
    train_ds = SentimentDataset(train_df, tokenizer, args.max_len)
    dev_ds = SentimentDataset(dev_df, tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- 模型 ----
    model = TransformerClassifier(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        n_classes=2,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_idx=tokenizer.pad_id,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}  (可训练: {trainable_params:,})")

    # ---- 优化器 & 调度器 ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps // 3, T_mult=1)

    # warmup 阶段：线性增长
    warmup_steps = int(total_steps * args.warmup_ratio)
    base_lr = args.lr

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return base_lr * step / max(warmup_steps, 1)
        return scheduler.get_last_lr()[0]  # scheduler 已在 step 后更新

    # ---- 训练循环 ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_dev_acc = 0.0
    best_epoch = 0
    global_step = 0
    patience = 3  # early stop: 连续 N 个 epoch dev acc 不涨就停
    no_improve_count = 0

    print(f"\n{'=' * 50}")
    print("  开始训练")
    print(f"{'=' * 50}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # warmup + cosine LR
            if global_step < warmup_steps:
                lr = base_lr * global_step / max(warmup_steps, 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            elif global_step == warmup_steps:
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=max(1, (total_steps - warmup_steps) // 2),
                    T_mult=1,
                )

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if global_step >= warmup_steps:
                scheduler.step()

            train_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch}/{args.epochs} | "
                    f"Step {batch_idx}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} | lr={current_lr:.2e}"
                )

        avg_train_loss = train_loss / len(train_loader)
        dev_acc = evaluate(model, dev_loader, device)
        print(
            f"\n  Epoch {epoch} 完成 | "
            f"train_loss={avg_train_loss:.4f} | dev_acc={dev_acc:.4f}"
        )

        # 保存最佳模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(
                model,
                tokenizer,
                CKPT_DIR,
                "best.pt",
                epoch=epoch,
                dev_acc=dev_acc,
                train_loss=avg_train_loss,
            )
            print(f"  >>> 新最佳模型 (dev_acc={dev_acc:.4f})")
        else:
            no_improve_count += 1
            print(f"  未提升 ({no_improve_count}/{patience})")

        # Early stopping
        if not args.no_early_stop and no_improve_count >= patience:
            print(f"\n  Early stop: 连续 {patience} epoch 无提升，终止训练。")
            break

    # 保存最后模型
    save_checkpoint(
        model,
        tokenizer,
        CKPT_DIR,
        "last.pt",
        epoch=epoch,
        dev_acc=dev_acc,
        train_loss=avg_train_loss,
    )

    print(f"\n{'=' * 50}")
    print(f"  训练结束 | 最佳 dev_acc={best_dev_acc:.4f} (epoch {best_epoch})")
    print(f"{'=' * 50}")

    # ---- 注意力热图 ----
    # 从 dev set 中选 3 个样本画热图
    print("\n生成注意力热图 ...")
    dev_samples = dev_df.sample(n=min(10, len(dev_df)), random_state=args.seed)
    # 优先选：1 正面、1 负面、1 长句
    pos_samples = dev_df[dev_df["label"] == 1]
    neg_samples = dev_df[dev_df["label"] == 0]
    heatmap_texts = []

    if len(pos_samples) > 0:
        row = pos_samples.sample(1, random_state=args.seed).iloc[0]
        heatmap_texts.append((row["text"], "positive"))
    if len(neg_samples) > 0:
        row = neg_samples.sample(1, random_state=args.seed + 1).iloc[0]
        heatmap_texts.append((row["text"], "negative"))
    # 长句
    dev_df["text_len"] = dev_df["text"].str.len()
    long_row = dev_df.nlargest(3, "text_len").sample(1, random_state=args.seed).iloc[0]
    heatmap_texts.append((long_row["text"], "long"))

    # 加载最佳模型画热图
    best_ckpt = CKPT_DIR / "best.pt"
    if best_ckpt.exists():
        from src.model import load_for_eval

        model, _ = load_for_eval(str(best_ckpt))
        model = model.to(device)
        plot_attention_heatmaps(model, tokenizer, heatmap_texts, device, FIG_DIR)
    else:
        print("  (无 best.pt，跳过热图生成)")

    # ---- 自检提示 ----
    print(f"\n运行自检查看是否达标: python eval/run.py")


if __name__ == "__main__":
    main()
