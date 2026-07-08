"""Train mini-GPT on data/train.txt with BPE + next-token prediction.

用法：
    python data/download.py              # 准备 train/dev
    python train.py                      # 默认唐诗 quick-start（默认开启 wandb）
    python train.py --epochs 30          # 更多 epoch
    python train.py --device cuda        # GPU
    python train.py --no-wandb           # 不上传 wandb
    WANDB_MODE=offline python train.py   # 离线记录，之后 wandb sync
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.model import MiniGPT, MiniGPTConfig, save_checkpoint
from src.tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train mini-GPT")
    p.add_argument("--train-file", type=str, default=str(ROOT / "data" / "train.txt"))
    p.add_argument("--dev-file", type=str, default=str(ROOT / "data" / "dev.txt"))
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / "ckpt"))
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-every", type=int, default=5, help="每 N 个 epoch 打印一次生成样例")
    p.add_argument("--log-every", type=int, default=10, help="每 N step 记录一次 wandb")
    p.add_argument("--bpe-max-mb", type=float, default=2,
                   help="BPE 语料字符上限（按文档读取，默认约 2MB 文本）")
    p.add_argument("--bpe-max-docs", type=int, default=2000,
                   help="BPE 最多用多少篇文档")
    p.add_argument("--chunk-kb", type=int, default=8,
                   help="训练时每次随机读取的文本块（KB），越小 encode 越快，默认 8KB")
    p.add_argument("--steps-per-epoch", type=int, default=2000,
                   help="大语料每 epoch 最多训练多少 step（默认 2000，避免 1 epoch 跑几万步）")
    p.add_argument("--dev-max-tokens", type=int, default=4096,
                   help="dev 困惑度最多用多少 token（与 eval 一致）")
    p.add_argument("--reuse-tokenizer", action="store_true",
                   help="若 ckpt/tokenizer.json 已存在则跳过 BPE 训练")
    # wandb
    p.add_argument("--no-wandb", action="store_true", help="禁用 Weights & Biases 日志")
    p.add_argument("--wandb-project", type=str, default="llm-beginner-p2-mini-gpt")
    p.add_argument("--wandb-run-name", type=str, default=None, help="run 名称，默认自动生成")
    p.add_argument("--wandb-entity", type=str, default=None, help="wandb 团队/用户名，默认用已登录账号")
    return p.parse_args()


def init_wandb(args: argparse.Namespace, extra: dict | None = None):
    """初始化 wandb；未安装、--no-wandb 或初始化失败时返回 None。"""
    if args.no_wandb:
        print("[wandb] 已禁用（--no-wandb）")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] 未安装，跳过。可执行: pip install wandb")
        return None

    config = {**vars(args), **(extra or {})}
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=config,
        )
        print(f"[wandb] 已连接: {run.url}")
        return run
    except Exception as e:
        print(f"[wandb] 初始化失败，继续训练（无日志）: {e}")
        return None


def wandb_log(metrics: dict, step: int | None = None) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def wandb_finish(summary: dict | None = None) -> None:
    try:
        import wandb

        if wandb.run is not None:
            if summary:
                for k, v in summary.items():
                    wandb.run.summary[k] = v
            wandb.finish()
    except ImportError:
        pass


def load_dataset_info() -> dict:
    path = ROOT / "data" / "dataset_info.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text_head(path: Path, max_bytes: int) -> str:
    """读取文件开头若干字节（UTF-8 安全截断）。"""
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="ignore")
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="ignore")


def read_bpe_corpus(path: Path, max_chars: int = 2_000_000, max_docs: int = 5000) -> str:
    """按文档流式读取，供 BPE 训练用，避免一次加载 GB 级文本。"""
    parts: list[str] = []
    total = 0
    doc_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if doc_lines:
                    doc = "".join(doc_lines).strip()
                    doc_lines = []
                    if doc:
                        parts.append(doc)
                        total += len(doc)
                        if total >= max_chars or len(parts) >= max_docs:
                            break
            else:
                doc_lines.append(line)
        if doc_lines and total < max_chars and len(parts) < max_docs:
            doc = "".join(doc_lines).strip()
            if doc:
                parts.append(doc)
    return "\n\n".join(parts)


def read_random_document(path: Path, max_bytes: int = 8192) -> str:
    """随机读一篇短文档（约几 KB），避免每次 encode 数 MB 文本。"""
    size = path.stat().st_size
    if size == 0:
        return ""
    max_bytes = min(max_bytes, size)
    start = random.randint(0, max(0, size - max_bytes))
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="ignore")
    if start > 0:
        _, _, text = text.partition("\n\n")
    doc = text.split("\n\n")[0].strip()
    return doc or text.strip()


def encode_file_limited(tokenizer: BPETokenizer, path: Path, max_bytes: int, max_tokens: int) -> list[int]:
    text = read_text_head(path, max_bytes)
    return tokenizer.encode(text)[:max_tokens]


def _make_block(ids: list[int], block_size: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """保证产出 block_size+1 个 token，空输入用 pad 填充。"""
    need = block_size + 1
    if not ids:
        ids = [pad_id]
    if len(ids) < need:
        ids = (ids * ((need // len(ids)) + 1))[:need]
    if len(ids) < need:
        ids = ids + [pad_id] * (need - len(ids))
    start = 0 if len(ids) <= need else random.randint(0, len(ids) - need)
    chunk = ids[start : start + need]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y


class StreamingTokenBlockDataset(Dataset):
    """从大文本文件随机读短文档、现场 encode，不把全语料 token 化进内存。"""

    def __init__(
        self,
        path: Path,
        tokenizer: BPETokenizer,
        block_size: int,
        chunk_bytes: int,
        steps_per_epoch: int,
        batch_size: int,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chunk_bytes = chunk_bytes
        self.n_blocks = steps_per_epoch * batch_size

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int):
        pad_id = self.tokenizer.pad_id
        for _ in range(8):
            text = read_random_document(self.path, self.chunk_bytes)
            if text.strip():
                ids = self.tokenizer.encode(text)
                if ids:
                    return _make_block(ids, self.block_size, pad_id)
        return _make_block([], self.block_size, pad_id)


class TokenBlockDataset(Dataset):
    """Random contiguous blocks from tokenized corpus."""

    def __init__(self, token_ids: list[int], block_size: int):
        self.data = token_ids
        self.block_size = block_size
        self.n_blocks = max(1, len(token_ids) // block_size)

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int):
        start = random.randint(0, max(0, len(self.data) - self.block_size - 1))
        chunk = self.data[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


@torch.no_grad()
def compute_ppl(model: MiniGPT, token_ids: list[int], device: str) -> float:
    model.eval()
    block = model.block_size
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(token_ids) - 1), block):
        window = token_ids[i : i + block + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk)
        nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        n_tok += chunk.size(1) - 1
    return math.exp(nll / n_tok) if n_tok else float("inf")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用（驱动过旧或未检测到 GPU），自动改用 CPU")
        device = "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_file)
    dev_path = Path(args.dev_file)
    if not train_path.exists():
        sys.exit(f"[错误] 找不到 {train_path}，请先运行 data/download.py")

    train_mb = train_path.stat().st_size / (1024 ** 2)
    tok_path = ckpt_dir / "tokenizer.json"

    if args.reuse_tokenizer and tok_path.exists():
        print(f"Reusing tokenizer -> {tok_path}")
        tokenizer = BPETokenizer.from_pretrained(tok_path)
    else:
        bpe_chars = int(args.bpe_max_mb * 1024 * 1024)
        print(f"Training BPE tokenizer (≤{args.bpe_max_mb}MB / ≤{args.bpe_max_docs} docs, 共 {train_mb:.1f}MB) ...")
        bpe_text = read_bpe_corpus(train_path, max_chars=bpe_chars, max_docs=args.bpe_max_docs)
        print(f"  实际读取 {len(bpe_text)/1e6:.2f}M chars, {bpe_text.count(chr(10))+1} lines")
        tokenizer = BPETokenizer.train(bpe_text, vocab_size=args.vocab_size, max_words=2000)
        tokenizer.save(tok_path)
        print(f"  vocab_size={tokenizer.vocab_size}, saved -> {tok_path}")

    dev_bytes = int(min(10, args.bpe_max_mb) * 1024 * 1024)
    dev_ids = encode_file_limited(tokenizer, dev_path, dev_bytes, args.dev_max_tokens) if dev_path.exists() else []
    print(f"  train file: {train_mb:.1f}MB (streaming) | dev tokens for ppl: {len(dev_ids)}")

    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")

    dataset_info = load_dataset_info()
    init_wandb(
        args,
        extra={
            "vocab_size": tokenizer.vocab_size,
            "train_file_mb": round(train_mb, 1),
            "dev_tokens": len(dev_ids),
            "total_params": n_params,
            "dataset": dataset_info.get("dataset", "unknown"),
            "ppl_threshold": dataset_info.get("ppl_threshold"),
        },
    )

    chunk_bytes = args.chunk_kb * 1024
    if train_mb > 100:
        dataset = StreamingTokenBlockDataset(
            train_path, tokenizer, args.block_size, chunk_bytes,
            steps_per_epoch=args.steps_per_epoch, batch_size=args.batch_size,
        )
        n_steps = len(dataset) // args.batch_size
        print(f"  使用 StreamingTokenBlockDataset（doc≤{args.chunk_kb}KB，{n_steps} steps/epoch）")
    else:
        train_text = train_path.read_text(encoding="utf-8")
        train_ids = tokenizer.encode(train_text)
        dataset = TokenBlockDataset(train_ids, args.block_size)
        print(f"  小语料模式：train tokens={len(train_ids)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    sched_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup), eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup])

    best_ppl = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if global_step % args.log_every == 0:
                wandb_log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        avg_loss = total_loss / len(loader)
        dev_ppl = compute_ppl(model, dev_ids, device) if dev_ids else float("nan")
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, dev_ppl={dev_ppl:.2f}")
        wandb_log(
            {
                "train/epoch_loss": avg_loss,
                "dev/ppl": dev_ppl,
                "epoch": epoch,
            },
            step=global_step,
        )

        if dev_ids and dev_ppl < best_ppl:
            best_ppl = dev_ppl
            save_checkpoint(ckpt_dir / "best.pt", model, tok_path, extra={"dev_ppl": dev_ppl})
            print(f"  -> saved best.pt (dev_ppl={dev_ppl:.2f})")
            wandb_log({"dev/best_ppl": best_ppl}, step=global_step)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            prompt = "床前明月光" if dataset_info.get("dataset") != "tinystories" else "Once upon a time"
            prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            out_ids = model.generate(prompt_ids, max_new_tokens=32, top_p=0.9, temperature=0.8)
            sample_text = tokenizer.decode(out_ids.tolist())
            print(f"  sample: {sample_text}")
            wandb_log({"sample/generation": sample_text, "epoch": epoch}, step=global_step)

    save_checkpoint(ckpt_dir / "last.pt", model, tok_path, extra={"dev_ppl": dev_ppl})
    print(f"\nDone. Best dev_ppl={best_ppl:.2f}. Run: python eval/run.py")
    wandb_finish(summary={"best_dev_ppl": best_ppl, "final_dev_ppl": dev_ppl})


if __name__ == "__main__":
    main()
