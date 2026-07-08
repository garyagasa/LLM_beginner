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


class LoRALinear(nn.Module):
    """在线性层旁路注入 LoRA 低秩分支。

    Args:
        base: 被替换的 nn.Linear（权重将被冻结）
        r:    LoRA rank
        alpha: LoRA scaling 系数，实际缩放 = alpha / r
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(base.in_features, r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(r, base.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_A @ self.lora_B)

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


def _get_parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    alpha: float = 16,
) -> nn.Module:
    """在 model 中把匹配的 nn.Linear 替换为 LoRALinear。"""
    to_replace: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _module_name_matches(name, target_modules):
            to_replace.append(name)

    for name in to_replace:
        parent, child_name = _get_parent_and_child(model, name)
        old_linear = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old_linear, r, alpha))

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """将 LoRA 权重合并回原 Linear，并恢复为普通 nn.Linear。"""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        parent, child_name = _get_parent_and_child(model, name)

        delta = module.scaling * (module.lora_A @ module.lora_B).T
        merged_weight = module.base.weight.data + delta

        new_linear = nn.Linear(
            module.base.in_features,
            module.base.out_features,
            bias=module.base.bias is not None,
            device=module.base.weight.device,
            dtype=module.base.weight.dtype,
        )
        new_linear.weight.data = merged_weight
        if module.base.bias is not None:
            new_linear.bias.data = module.base.bias.data

        setattr(parent, child_name, new_linear)

    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只导出 LoRA 相关参数（用于保存 ckpt/sft 与 ckpt/dpo）。"""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state_dict: dict) -> None:
    """加载 LoRA 权重到已 inject_lora 的 model。"""
    model.load_state_dict(state_dict, strict=False)
