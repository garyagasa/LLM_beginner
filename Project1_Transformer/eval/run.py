"""任务一自检：单元测试 + 文本分类准确率 + 热图检查。

约定学生实现见同目录 ../README.md「实现约定」一节。
本脚本运行后会输出每项测试结果，并把结果写入 eval/result.json，可附在提交里。

用法：
    python eval/run.py                  # 跑全部测试
    python eval/run.py --list           # 列出所有测试
    python eval/run.py M1               # 只跑 M1（不区分大小写）
    python eval/run.py M2 M4            # 跑多个里程碑
    python eval/run.py mha_forward      # 按测试名筛选
    python eval/run.py --only causal_mask attention_heatmaps
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests

ACC_PASS = 0.80       # 文本分类准确率通过线
ACC_BASELINE = 0.85   # 参考基线，仅供对照、不参与判定
HEATMAP_MIN = 3       # M5：至少几张热图


# ---------------------------------------------------------------------------
# M1：scaled_dot_product_attention 数值正确
# ---------------------------------------------------------------------------

def test_attention_correctness():
    """手写 attention 应与 torch 官方实现数值一致（误差 < 1e-5）。"""
    from src.attention import scaled_dot_product_attention

    torch.manual_seed(0)
    B, H, T, D = 2, 4, 8, 16
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)

    out_student = scaled_dot_product_attention(Q, K, V)
    out_ref = F.scaled_dot_product_attention(Q, K, V)

    diff = (out_student - out_ref).abs().max().item()
    return {
        "test": "attention_correctness",
        "dod": "M1",
        "pass": diff < 1e-5,
        "max_abs_diff": diff,
    }


# ---------------------------------------------------------------------------
# M2：MultiHeadAttention + TransformerBlock 前向与形状
# ---------------------------------------------------------------------------

def test_mha_forward():
    """MultiHeadAttention 前向不报错，输出形状 (B, T, d_model)。"""
    from src.attention import MultiHeadAttention

    torch.manual_seed(0)
    B, T, d_model, n_heads = 2, 10, 128, 4
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)
    out = mha(x)

    shape_ok = out.shape == (B, T, d_model)
    finite_ok = bool(torch.isfinite(out).all())
    return {
        "test": "mha_forward",
        "dod": "M2",
        "pass": shape_ok and finite_ok,
        "output_shape": list(out.shape),
        "expected_shape": [B, T, d_model],
    }


def test_mha_padding_mask():
    """padding mask 下，PAD 位置不应从有效 token 吸收注意力。"""
    from src.attention import MultiHeadAttention

    torch.manual_seed(0)
    B, T, d_model, n_heads = 1, 6, 64, 4
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(B, T, d_model)

    # 最后 2 个位置视为 PAD：(B, 1, 1, T)
    pad_mask = torch.zeros(B, 1, 1, T, dtype=torch.bool)
    pad_mask[:, :, :, -2:] = True

    out_valid = mha(x, mask=pad_mask)
    x2 = x.clone()
    x2[:, -2:, :] = 999.0
    out_valid2 = mha(x2, mask=pad_mask)

    # 非 PAD 位置输出不应因 PAD 位输入改变而变化
    leaked = (out_valid[:, :-2] - out_valid2[:, :-2]).abs().max().item()
    return {
        "test": "mha_padding_mask",
        "dod": "M2",
        "pass": leaked < 1e-5,
        "leaked_diff": leaked,
    }


def test_transformer_block_forward():
    """TransformerBlock 前向不报错，输出形状与输入一致。"""
    from src.block import TransformerBlock

    torch.manual_seed(0)
    B, T, d_model, n_heads, d_ff = 2, 8, 128, 4, 512
    block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
    block.eval()
    x = torch.randn(B, T, d_model)
    out = block(x)

    shape_ok = out.shape == x.shape
    finite_ok = bool(torch.isfinite(out).all())
    return {
        "test": "transformer_block_forward",
        "dod": "M2",
        "pass": shape_ok and finite_ok,
        "output_shape": list(out.shape),
        "input_shape": list(x.shape),
    }


# ---------------------------------------------------------------------------
# M3：分类器 dev 准确率
# ---------------------------------------------------------------------------

def test_classifier_accuracy():
    """跑学生训练好的 checkpoint 在 ChnSentiCorp dev set 上的准确率。"""
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        return {
            "test": "classifier_accuracy",
            "dod": "M3",
            "pass": None,
            "skip": "ckpt/best.pt 不存在，先完成训练：python train.py",
        }

    try:
        from src.model import load_for_eval
    except ImportError as e:
        return {
            "test": "classifier_accuracy",
            "dod": "M3",
            "pass": False,
            "error": f"src/model.py 未导出 load_for_eval: {e}",
        }

    dev_path = ROOT / "data" / "validation.parquet"
    if not dev_path.exists():
        return {
            "test": "classifier_accuracy",
            "dod": "M3",
            "pass": None,
            "skip": "data/validation.parquet 不存在，先跑 data/download.py",
        }

    import pandas as pd

    dev = pd.read_parquet(dev_path)
    model, tokenize_fn = load_for_eval(str(ckpt))
    model.eval()

    correct = 0
    total = len(dev)
    with torch.no_grad():
        for _, row in dev.iterrows():
            ids = tokenize_fn(row["text"])
            logits = model(ids.unsqueeze(0))
            pred = int(logits.argmax(dim=-1).item())
            if pred == int(row["label"]):
                correct += 1

    acc = correct / total
    return {
        "test": "classifier_accuracy",
        "dod": "M3",
        "pass": acc >= ACC_PASS,
        "accuracy": round(acc, 4),
        "threshold": ACC_PASS,
        "baseline_reference": ACC_BASELINE,
    }


# ---------------------------------------------------------------------------
# M4：causal mask + toy 语言模型前向
# ---------------------------------------------------------------------------

def test_causal_mask():
    """causal mask 下，位置 i 的输出不应被位置 j>i 的 V 改动影响。"""
    from src.attention import scaled_dot_product_attention

    torch.manual_seed(0)
    B, H, T, D = 1, 1, 5, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # True = 被屏蔽

    out = scaled_dot_product_attention(Q, K, V, mask=mask)
    V2 = V.clone()
    V2[:, :, -1] = 999.0  # 改最后一位
    out2 = scaled_dot_product_attention(Q, K, V2, mask=mask)

    leaked = (out[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    return {
        "test": "causal_mask",
        "dod": "M4",
        "pass": leaked < 1e-6,
        "leaked_diff": leaked,
    }


def test_toy_lm_forward():
    """最小 toy LM：Embedding + causal MHA + Linear，验证能前向且不泄漏未来信息。"""
    from src.attention import MultiHeadAttention

    class ToyLM(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, n_heads: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            T = ids.size(1)
            causal = torch.triu(torch.ones(T, T, device=ids.device), diagonal=1).bool()
            causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            x = self.embed(ids)
            x = self.attn(x, mask=causal)
            return self.head(x)

    torch.manual_seed(0)
    lm = ToyLM(vocab_size=50, d_model=64, n_heads=4)
    lm.eval()
    ids = torch.randint(0, 50, (2, 12))
    logits = lm(ids)

    shape_ok = logits.shape == (2, 12, 50)
    finite_ok = bool(torch.isfinite(logits).all())

    # 改最后一个 token 的 embedding 输入，前面位置的 logits 不应变
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % 50
    logits2 = lm(ids2)
    leaked = (logits[:, :-1] - logits2[:, :-1]).abs().max().item()

    return {
        "test": "toy_lm_forward",
        "dod": "M4",
        "pass": shape_ok and finite_ok and leaked < 1e-5,
        "output_shape": list(logits.shape),
        "leaked_diff": leaked,
    }


# ---------------------------------------------------------------------------
# M5：注意力热图文件
# ---------------------------------------------------------------------------

def test_attention_heatmaps():
    """figures/ 下应有 >= 3 张注意力热图（train.py 训练完会自动生成）。"""
    fig_dir = ROOT / "figures"
    if not fig_dir.exists():
        return {
            "test": "attention_heatmaps",
            "dod": "M5",
            "pass": None,
            "skip": "figures/ 不存在，先完成训练：python train.py",
        }

    pngs = sorted(fig_dir.glob("*.png"))
    n = len(pngs)
    return {
        "test": "attention_heatmaps",
        "dod": "M5",
        "pass": n >= HEATMAP_MIN,
        "count": n,
        "required": HEATMAP_MIN,
        "files": [p.name for p in pngs],
    }


# ---------------------------------------------------------------------------
# 注册表 & CLI
# ---------------------------------------------------------------------------

TestFn = Callable[[], dict]

TEST_REGISTRY: list[tuple[str, str, TestFn]] = [
    ("M1", "attention_correctness", test_attention_correctness),
    ("M2", "mha_forward", test_mha_forward),
    ("M2", "mha_padding_mask", test_mha_padding_mask),
    ("M2", "transformer_block_forward", test_transformer_block_forward),
    ("M3", "classifier_accuracy", test_classifier_accuracy),
    ("M4", "causal_mask", test_causal_mask),
    ("M4", "toy_lm_forward", test_toy_lm_forward),
    ("M5", "attention_heatmaps", test_attention_heatmaps),
]

_NAME_TO_FN: dict[str, TestFn] = {name: fn for _, name, fn in TEST_REGISTRY}
_DOD_TO_NAMES: dict[str, list[str]] = {}
for dod, name, _ in TEST_REGISTRY:
    _DOD_TO_NAMES.setdefault(dod, []).append(name)


def list_tests() -> None:
    print("可用测试（python eval/run.py <目标>）：\n")
    print(f"{'DoD':<4} {'测试名':<28} 说明")
    print("-" * 72)
    desc = {
        "attention_correctness": "与官方 attention 数值一致",
        "mha_forward": "MultiHeadAttention 形状与前向",
        "mha_padding_mask": "padding mask 不泄漏",
        "transformer_block_forward": "TransformerBlock 形状与前向",
        "classifier_accuracy": f"dev 准确率 ≥ {ACC_PASS}",
        "causal_mask": "causal mask 不泄漏未来 V",
        "toy_lm_forward": "toy LM + causal mask 前向",
        "attention_heatmaps": f"figures/ 下 ≥ {HEATMAP_MIN} 张热图",
    }
    for dod, name, fn in TEST_REGISTRY:
        print(f"{dod:<4} {name:<28} {desc.get(name, fn.__doc__ or '')}")
    print("\n示例：")
    print("  python eval/run.py              # 全部")
    print("  python eval/run.py M1           # 只测 attention")
    print("  python eval/run.py M2 M4        # 测 block + causal")
    print("  python eval/run.py mha_forward  # 按测试名")


def resolve_tests(targets: list[str]) -> list[TestFn]:
    """将 CLI 目标（M1 / 测试名）解析为测试函数列表，保持注册顺序。"""
    if not targets:
        return [fn for _, _, fn in TEST_REGISTRY]

    selected_names: list[str] = []
    for raw in targets:
        key = raw.strip()
        upper = key.upper()
        if upper in _DOD_TO_NAMES:
            for name in _DOD_TO_NAMES[upper]:
                if name not in selected_names:
                    selected_names.append(name)
            continue
        if key in _NAME_TO_FN:
            if key not in selected_names:
                selected_names.append(key)
            continue
        raise SystemExit(
            f"未知测试目标: {raw!r}\n"
            f"用 python eval/run.py --list 查看 M1–M5 或测试名"
        )

    return [_NAME_TO_FN[n] for n in selected_names]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project1 Transformer 自检（M1–M5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="不带参数时运行全部测试；可用 M1–M5 或具体测试名筛选。",
    )
    p.add_argument(
        "targets",
        nargs="*",
        metavar="TARGET",
        help="M1–M5 或测试名（如 attention_correctness）",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="TARGET",
        help="与位置参数相同，显式指定要跑的测试",
    )
    p.add_argument("--list", action="store_true", help="列出所有测试后退出")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        list_tests()
        sys.exit(0)

    targets = args.targets or args.only or []
    tests = resolve_tests(targets)
    results = run_tests(tests, ROOT)

    failed = [r for r in results if r.get("pass") is False]
    if failed:
        sys.exit(1)
