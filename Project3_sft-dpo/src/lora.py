"""M1：手写 LoRA —— 低秩矩阵注入、forward、只更新 LoRA 参数。

---- 接口约定（eval/run.py 自检依赖）----
  inject_lora(model, target_modules, r, alpha) -> model
    在 model 的 target_modules（如 q_proj / v_proj）上注入 LoRA，冻结原权重。
  merge_lora(model) -> model
    将 LoRA 权重合并回原 Linear，便于推理导出。

LoRA 公式（对 Linear W: in×out）：
  y = W x + (alpha / r) · B(A x)
  A: in×r（Kaiming 初始化），B: r×out（零初始化），原 W 冻结。
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """在线性层旁路注入 LoRA 低秩分支。

    Args:
        base: 被替换的 nn.Linear（权重将被冻结）
        r:    LoRA rank
        alpha: LoRA scaling 系数，实际缩放 = alpha / r
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        # TODO: 保存 base、r、scaling；创建 lora_A (in_features × r) 与 lora_B (r × out_features)
        # TODO: 初始化 lora_A 用 kaiming_uniform_，lora_B 用 zeros_
        # TODO: 冻结 base.weight 和 base.bias（requires_grad=False）
        raise NotImplementedError("TODO: 实现 LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return base(x) + scaling * (x @ lora_A @ lora_B)
        # 提示：可用 F.linear(x, lora_A.T) 等价于 x @ lora_A
        raise NotImplementedError("TODO: 实现 LoRALinear.forward")

    @property
    def weight(self):
        """兼容 merge / save 时访问原权重。"""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def _module_name_matches(name: str, target_modules: Iterable[str]) -> bool:
    """判断模块全名是否以 target_modules 中任一后缀结尾。"""
    return any(name.endswith(f".{t}") or name == t for t in target_modules)


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: float = 16,
) -> nn.Module:
    """在 model 中把匹配的 nn.Linear 替换为 LoRALinear。

    Args:
        model: HuggingFace CausalLM（如 Qwen2.5-0.5B）
        target_modules: 要注入的模块名后缀，如 ["q_proj", "v_proj"]
        r: LoRA rank
        alpha: LoRA scaling

    Returns:
        注入 LoRA 后的 model（原地修改）

    实现步骤：
        1. 遍历 model.named_modules()，找到 name 匹配 target_modules 的 nn.Linear
        2. 用 LoRALinear(base, r, alpha) 替换（通过 parent  setattr）
        3. 确保只有 LoRA 参数 requires_grad=True
    """
    # ---- 你的代码开始 ----
    # TODO: 遍历并替换 Linear 层
    raise NotImplementedError("TODO: 实现 inject_lora")
    # ---- 你的代码结束 ----
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """将 LoRA 权重合并回原 Linear，并恢复为普通 nn.Linear。

    合并公式：W_merged = W + scaling * (lora_A @ lora_B).T
    （注意 Linear 权重形状为 out_features × in_features）

    Returns:
        合并后的 model（原地修改）
    """
    # ---- 你的代码开始 ----
    # TODO: 遍历 LoRALinear，计算 merged weight，替换回 nn.Linear
    raise NotImplementedError("TODO: 实现 merge_lora")
    # ---- 你的代码结束 ----
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只导出 LoRA 相关参数（用于保存 ckpt/sft 与 ckpt/dpo）。"""
    # TODO: 收集 name 中含 "lora_" 且 requires_grad 的参数
    raise NotImplementedError("TODO: 实现 lora_state_dict")


def load_lora_state_dict(model: nn.Module, state_dict: dict) -> None:
    """加载 LoRA 权重到已 inject_lora 的 model。"""
    # TODO: model.load_state_dict(state_dict, strict=False)
    raise NotImplementedError("TODO: 实现 load_lora_state_dict")
