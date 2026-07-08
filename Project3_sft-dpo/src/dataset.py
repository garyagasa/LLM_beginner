"""M3 辅助：SFT / DPO 数据集加载与预处理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from src.chat import build_labels, format_messages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_PATH = ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict]:
    """逐行读取 jsonl。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def moss_messages_from_record(record: dict) -> List[dict]:
    """将 MOSS jsonl 单条记录转为 messages 列表。

    MOSS 格式示例：
        {"conversation_id": "...", "chat": {"turn_1": {"Human": "...", "MOSS": "..."}, ...}}

    Returns:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    # TODO: 解析 record["chat"] 中的 turn_*，Human → user，MOSS → assistant
    raise NotImplementedError("TODO: 实现 moss_messages_from_record")


class SFTDataset(Dataset):
    """MOSS SFT 数据集：返回 input_ids / labels / attention_mask。"""

    def __init__(
        self,
        data_path: Path | str = DEFAULT_SFT_PATH,
        tokenizer=None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[List[dict]] = []

        # TODO: 读取 jsonl，解析为 messages，截断到 max_samples 条
        # TODO: 若文件不存在，打印 data/download.py 中的下载提示
        raise NotImplementedError("TODO: 实现 SFTDataset.__init__")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO:
        #   1. messages = self.samples[idx]
        #   2. text = format_messages(messages)
        #   3. enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        #   4. labels = build_labels(enc.input_ids[0], messages)
        #   5. return {"input_ids": ..., "labels": ..., "attention_mask": ...}
        raise NotImplementedError("TODO: 实现 SFTDataset.__getitem__")


class DPODataset(Dataset):
    """偏好数据集：每条含 prompt + chosen + rejected。"""

    def __init__(
        self,
        data_path: Path | str,
        tokenizer=None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[dict] = []

        # TODO: 加载 DPO 数据（如 hiyouga/DPO-En-Zh-20k 导出的 jsonl）
        # 每条至少包含：prompt, chosen, rejected（或等价字段名）
        raise NotImplementedError("TODO: 实现 DPODataset.__init__")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # TODO: tokenize prompt/chosen/rejected，返回 policy / ref model 需要的字段
        raise NotImplementedError("TODO: 实现 DPODataset.__getitem__")
