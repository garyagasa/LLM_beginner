"""M4：DPO 损失函数。

DPO 目标（单条样本）：
    L = -log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) )

其中 y_w = chosen，y_l = rejected，β 为温度系数。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """对 labels 位置计算平均 log prob（labels == -100 的位置跳过）。

    Args:
        logits: (B, T, V)
        labels: (B, T)，-100 为 mask
        attention_mask: 可选 (B, T)

    Returns:
        (B,) 每条样本的平均 log prob
    """
    # TODO: shift logits/labels 做 next-token prediction，gather 对应 token 的 log prob
    raise NotImplementedError("TODO: 实现 _sequence_log_probs")


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """计算 DPO loss 与辅助指标。

    Args:
        *_logps: shape (B,)，policy / ref 在 chosen/rejected 上的序列 log prob
        beta: DPO 温度

    Returns:
        loss: 标量
        metrics: {"reward_margin": ..., "accuracy": ...} 等
    """
    # ---- 你的代码开始 ----
    # TODO:
    #   pi_logratios = policy_chosen_logps - policy_rejected_logps
    #   ref_logratios = ref_chosen_logps - ref_rejected_logps
    #   logits = beta * (pi_logratios - ref_logratios)
    #   loss = -F.logsigmoid(logits).mean()
    raise NotImplementedError("TODO: 实现 dpo_loss")
    # ---- 你的代码结束 ----
