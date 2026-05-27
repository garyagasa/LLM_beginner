"""任务三：手写 LoRA（Low-Rank Adaptation）。

接口约定（eval/run.py 会自动检测）：
  - inject_lora(model, target_modules, r, alpha) -> model
  - merge_lora(model) -> model
"""

import torch
import torch.nn as nn
from typing import List


class LoRALinear(nn.Module):
    """包装一个 nn.Linear，添加低秩矩阵 A 和 B。"""

    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        """初始化。

        Args:
            linear: 原始线性层
            r: 低秩维度
            alpha: 缩放因子（实际缩放 = alpha / r）
        """
        # TODO: 1. 保存原始 linear，冻结其参数
        # TODO: 2. 初始化 A ~ N(0, 1/r)，B = zeros
        # TODO: 3. 计算 scaling = alpha / r
        super().__init__()
        raise NotImplementedError("TODO: 实现 LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        # TODO: 1. y_orig = self.linear(x)  (detached)
        # TODO: 2. y_lora = (x @ A^T) @ B^T * self.scaling
        # TODO: 3. return y_orig + y_lora
        raise NotImplementedError("TODO: 实现 LoRALinear.forward")


def inject_lora(model: nn.Module, target_modules: List[str], r: int, alpha: int) -> nn.Module:
    """向模型中注入 LoRA。

    Args:
        model: 基础模型（如 Qwen2.5-0.5B）
        target_modules: 需要注入的模块名列表，如 ["q_proj", "v_proj"]
        r: 低秩维度
        alpha: 缩放因子

    Returns:
        注入 LoRA 后的模型（只有 LoRA 参数可训）
    """
    # TODO: 1. 遍历 model 的所有 named_modules
    # TODO: 2. 对名称匹配 target_modules 的 nn.Linear，替换为 LoRALinear
    # TODO: 3. 冻结原始模型参数，只保留 LoRA 参数可训
    raise NotImplementedError("TODO: 实现 inject_lora")


def merge_lora(model: nn.Module) -> nn.Module:
    """将 LoRA 权重合并回原始线性层。

    Args:
        model: 已注入 LoRA 的模型

    Returns:
        合并后的模型（LoRA 权重已合入，不再有 LoRALinear）
    """
    # TODO: 1. 遍历所有 LoRALinear 模块
    # TODO: 2. W_merged = W_orig + B @ A * scaling
    # TODO: 3. 替换回普通 nn.Linear
    raise NotImplementedError("TODO: 实现 merge_lora")
