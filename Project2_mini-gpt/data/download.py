"""下载任务二训练数据，并生成 train.txt / dev.txt 自检输入。

用法：
  python data/download.py                         # 默认唐诗（quick-start）
  python data/download.py --dataset poetry        # 唐诗（quick-start）
  python data/download.py --dataset tinystories   # TinyStories（英文故事语料，约百 MB 量级）
  python data/download.py --dataset skypile       # 默认 10 万条（约 280MB）
  python data/download.py --dataset skypile --num-samples 5000000   # 500 万条（约 13GB）
  python data/download.py --dataset skypile --max-local           # 尽量下满本地可用空间（仍非全量 620GB）
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
ROOT = DATA_DIR.parent.parent  # llm-beginner 仓库根
SKYPILE_FULL_GB = 620          # 官方全量约 620GB / 150B tokens
BYTES_PER_SAMPLE = 2680        # 经验值：100k 条约 268MB
RESERVE_GB = 30                # 留给 ckpt / 缓存


def write_dataset_info(dataset, ppl_threshold, extra=None):
    info = {
        "dataset": dataset,
        "train": "train.txt",
        "dev": "dev.txt",
        "ppl_threshold": ppl_threshold,
    }
    if extra:
        info.update(extra)
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text_splits(text, dataset, ppl_threshold, dev_ratio=0.1):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 100:
        sys.exit(f"[错误] {dataset} 文本过短，无法切分 train/dev")
    split_at = max(1, int(len(text) * (1 - dev_ratio)))
    (DATA_DIR / "train.txt").write_text(text[:split_at], encoding="utf-8")
    (DATA_DIR / "dev.txt").write_text(text[split_at:], encoding="utf-8")
    write_dataset_info(dataset, ppl_threshold)
    print(f"已生成 train.txt / dev.txt（{dataset}，dev_ratio={dev_ratio}）")


def get_poetry():
    src = ROOT / "poetryFromTang.txt"
    if not src.exists():
        sys.exit(f"[错误] 找不到 {src}（应在仓库根）")
    dst = DATA_DIR / "poetry.txt"
    shutil.copy(src, dst)
    write_text_splits(src.read_text(encoding="utf-8"),
                      dataset="poetry", ppl_threshold=50)
    print(f"已拷贝 {src.name} -> {dst.name}")


def write_hf_text_split(split, out_path):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in split:
            text = str(row.get("text", "")).strip()
            if text:
                f.write(text)
                f.write("\n\n")


def get_tinystories():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets pyarrow")
    print("下载 roneneldan/TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "cache"))
    for split in ds.keys():
        out = DATA_DIR / f"tinystories-{split}.parquet"
        ds[split].to_parquet(str(out))
        print(f"  {split}: {len(ds[split])} -> {out.name}")

    train_split = ds["train"]
    dev_split = ds["validation"] if "validation" in ds else ds["train"].select(range(1000))
    write_hf_text_split(train_split, DATA_DIR / "train.txt")
    write_hf_text_split(dev_split, DATA_DIR / "dev.txt")
    write_dataset_info("tinystories", ppl_threshold=10,
                       extra={"note": "TinyStories 是英文故事语料；中文训练可换用 SkyPile 子集或其他中文语料。"})
    print("已额外生成 train.txt / dev.txt，供 eval/run.py 计算困惑度。")


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _estimate_skypile_samples(num_samples: int | None, max_gb: float | None, max_local: bool) -> int:
    if max_local:
        avail = _free_gb(DATA_DIR) - RESERVE_GB
        cap = min(avail, SKYPILE_FULL_GB - 1)
        if cap < SKYPILE_FULL_GB:
            print(f"[提示] SkyPile 全量约 {SKYPILE_FULL_GB}GB，当前可用 {avail:.0f}GB（已预留 {RESERVE_GB}GB）")
            print(f"       无法下载全量，将按磁盘上限抽取约 {cap:.0f}GB 子集\n")
        else:
            print(f"[提示] 将尝试下载接近全量（约 {SKYPILE_FULL_GB}GB），耗时可能数小时\n")
        max_gb = max(1.0, cap)
    if max_gb is not None:
        est = int(max_gb * (1024 ** 3) / BYTES_PER_SAMPLE)
        print(f"[auto] 目标体积约 {max_gb:.1f}GB -> 预计 {est:,} 条")
        return est
    return num_samples or 100000


def get_skypile(
    num_samples: int | None = 100000,
    dev_ratio: float = 0.1,
    max_gb: float | None = None,
    max_local: bool = False,
):
    """SkyPile-150B 全量约 620GB；用 streaming 抽子集，直接流式写入 train/dev。"""
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets")

    num_samples = _estimate_skypile_samples(num_samples, max_gb, max_local)
    train_n = int(num_samples * (1 - dev_ratio))
    train_path = DATA_DIR / "train.txt"
    dev_path = DATA_DIR / "dev.txt"

    print(f"streaming 下载 Skywork/SkyPile-150B 前 {num_samples:,} 条 "
          f"（train≈{train_n:,}，dev≈{num_samples - train_n:,}）...")

    ds = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
    valid = 0
    with train_path.open("w", encoding="utf-8", newline="\n") as tf, \
         dev_path.open("w", encoding="utf-8", newline="\n") as df:
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            out = tf if valid < train_n else df
            out.write(text)
            out.write("\n\n")
            valid += 1
            if valid % 10000 == 0:
                print(f"  已写入 {valid:,}/{num_samples:,} 条 "
                      f"（train {min(valid, train_n):,} / dev {max(0, valid - train_n):,}）...")

    if valid == 0:
        sys.exit("[错误] 未获取到任何文本，请检查网络或 HF 镜像设置")

    train_mb = train_path.stat().st_size / (1024 ** 2)
    dev_mb = dev_path.stat().st_size / (1024 ** 2)
    extra = {
        "note": f"SkyPile 子集 {valid:,} 条（全量约 {SKYPILE_FULL_GB}GB / 2.33 亿网页）",
        "num_samples": valid,
    }
    write_dataset_info("skypile", ppl_threshold=100, extra=extra)
    print(f"完成：有效 {valid:,} 条 | train {train_mb:.1f}MB + dev {dev_mb:.1f}MB -> {DATA_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["poetry", "tinystories", "skypile"],
                    default="poetry")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="skypile 抽样条数（默认 100000；与 --max-gb/--max-local 互斥时以后者为准）")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="skypile 目标体积上限（GB），自动换算条数")
    ap.add_argument("--max-local", action="store_true",
                    help="skypile 尽量用满本地可用磁盘（仍受 620GB 全量上限约束）")
    ap.add_argument("--dev-ratio", type=float, default=0.1,
                    help="dev 集占比（默认 0.1）")
    args = ap.parse_args()
    if args.dataset == "skypile":
        get_skypile(
            num_samples=args.num_samples or 100000,
            dev_ratio=args.dev_ratio,
            max_gb=args.max_gb,
            max_local=args.max_local,
        )
    else:
        {"poetry": get_poetry, "tinystories": get_tinystories}[args.dataset]()


if __name__ == "__main__":
    main()
