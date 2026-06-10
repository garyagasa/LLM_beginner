"""下载 ChnSentiCorp 中文情感分类数据集到当前目录。

依赖：pip install datasets pyarrow

默认从 Hugging Face 下载。如果国内访问 HF 不稳定：
  - 方案一（推荐）：设置环境变量 HF_ENDPOINT=https://hf-mirror.com
  - 方案二：用 ModelScope（见脚本末尾提示）
"""
import os
import sys
from pathlib import Path

try:
    import ssl
    import certifi

    def _patched_create_default_context(*args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(cafile=certifi.where())
        return ctx
    ssl.create_default_context = _patched_create_default_context
except Exception:
    pass  # 修复失败也继续
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 如下载缓慢，可先 export HF_ENDPOINT=https://hf-mirror.com 再重试\n")

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] 缺少依赖：pip install datasets pyarrow")

    # seamew/ChnSentiCorp 使用旧式 loading script，datasets>=4.0 已移除支持。
    # 改用同数据的 Parquet 镜像（与 seamew 版本一致：9600/1200/1200）。
    dataset_id = "lansinuote/ChnSentiCorp"
    print(f"正在下载 {dataset_id} ...")
    ds = load_dataset(dataset_id, cache_dir=str(DATA_DIR / "cache"))

    for split in ds.keys():
        out = DATA_DIR / f"{split}.parquet"
        ds[split].to_parquet(str(out))
        print(f"  {split}: {len(ds[split])} 条 -> {out.name}")

    print(f"\n完成。数据保存在 {DATA_DIR}")
    print("\n--- ModelScope 备选下载 ---")
    print("如果你完全无法访问 HF：")
    print("  pip install modelscope")
    print("  python -c \"from modelscope.msdatasets import MsDataset; "
          "ds = MsDataset.load('ChnSentiCorp'); print(ds)\"")


if __name__ == "__main__":
    main()
