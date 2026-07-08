"""Weights & Biases 日志工具（SFT / DPO 训练脚本共用）。"""

from __future__ import annotations

import argparse


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
