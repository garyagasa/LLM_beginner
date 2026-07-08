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
    """对 labels 位置计算平均 log prob（labels == -100 的位置跳过）。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous().bool()
        mask = mask & shift_mask

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask.float()

    denom = mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / denom


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """计算 DPO loss 与辅助指标。"""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    metrics = {
        "reward_margin": (policy_chosen_logps - policy_rejected_logps).mean().item(),
        "accuracy": (logits > 0).float().mean().item(),
    }
    return loss, metrics
