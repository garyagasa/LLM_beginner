"""M3 辅助：SFT / DPO 数据集加载与预处理。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset

from src.chat import build_labels, format_messages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_PATH = ROOT / "data" / "moss-sft" / "moss-003-sft-no-tools.jsonl"
DEFAULT_DPO_PATH = ROOT / "data" / "dpo" / "train.jsonl"


def iter_jsonl(path: Path) -> Iterator[dict]:
    """逐行读取 jsonl。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _clean_moss_field(text: str, role: str) -> str:
    if role == "Human":
        text = re.sub(r"^<\|Human\|>:\s*", "", text)
        text = re.sub(r"<eoh>\s*$", "", text)
    elif role == "MOSS":
        text = re.sub(r"^<\|MOSS\|>:\s*", "", text)
        text = re.sub(r"<eom>\s*$", "", text)
    return text.strip()


def moss_messages_from_record(record: dict) -> List[dict]:
    """将 MOSS jsonl 单条记录转为 messages 列表。"""
    messages: List[dict] = []

    meta = record.get("meta_instruction", "").strip()
    if meta:
        messages.append({"role": "system", "content": meta})

    chat = record.get("chat", {})
    turn_keys = sorted(chat.keys(), key=lambda key: int(key.split("_")[1]))
    for turn_key in turn_keys:
        turn = chat[turn_key]
        if "Human" in turn:
            content = _clean_moss_field(turn["Human"], "Human")
            if content:
                messages.append({"role": "user", "content": content})
        if "MOSS" in turn:
            content = _clean_moss_field(turn["MOSS"], "MOSS")
            if content:
                messages.append({"role": "assistant", "content": content})

    return messages


def dpo_prompt_to_messages(prompt: str) -> List[dict]:
    """将 DPO 数据里的 [system]/[user] prompt 转为 messages。"""
    messages: List[dict] = []
    matches = list(re.finditer(r"\[(system|user)\]\n", prompt))
    for i, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        content = prompt[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """按 batch 内最长序列 padding。"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels, attention_mask = [], [], []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def dpo_pad_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """DPO batch padding（chosen / rejected 两套序列）。"""
    max_len = max(
        max(item["chosen_input_ids"].size(0), item["rejected_input_ids"].size(0))
        for item in batch
    )

    out: Dict[str, List[torch.Tensor]] = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "chosen_attention_mask": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "rejected_attention_mask": [],
    }

    for item in batch:
        for prefix in ("chosen", "rejected"):
            seq_len = item[f"{prefix}_input_ids"].size(0)
            pad_len = max_len - seq_len
            out[f"{prefix}_input_ids"].append(
                torch.cat([
                    item[f"{prefix}_input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
            )
            out[f"{prefix}_labels"].append(
                torch.cat([
                    item[f"{prefix}_labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )
            out[f"{prefix}_attention_mask"].append(
                torch.cat([
                    item[f"{prefix}_attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )

    return {key: torch.stack(tensors) for key, tensors in out.items()}


class SFTDataset(Dataset):
    """MOSS SFT 数据集：返回 input_ids / labels / attention_mask。"""

    def __init__(
        self,
        data_path: Path | str = DEFAULT_SFT_PATH,
        tokenizer=None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is None:
            raise ValueError("SFTDataset 需要传入 tokenizer")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[List[dict]] = []

        path = Path(data_path)
        if not path.exists():
            print(
                f"[错误] SFT 数据不存在: {path}\n"
                "请参考 README 或运行 python data/download.py 下载 MOSS 数据。"
            )
            sys.exit(1)

        for record in iter_jsonl(path):
            messages = moss_messages_from_record(record)
            if len(messages) >= 2:
                self.samples.append(messages)
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(f"未从 {path} 读取到有效 SFT 样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        messages = self.samples[idx]
        text = format_messages(messages)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]
        labels = build_labels(
            input_ids, messages, tokenizer=self.tokenizer, max_length=self.max_length
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class DPODataset(Dataset):
    """偏好数据集：每条含 prompt + chosen + rejected。"""

    def __init__(
        self,
        data_path: Path | str,
        tokenizer=None,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        if tokenizer is None:
            raise ValueError("DPODataset 需要传入 tokenizer")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[dict] = []

        path = Path(data_path)
        if not path.exists():
            print(
                f"[错误] DPO 数据不存在: {path}\n"
                "请准备包含 prompt/chosen/rejected 字段的 jsonl。"
            )
            sys.exit(1)

        for record in iter_jsonl(path):
            prompt = record.get("prompt", "").strip()
            chosen = record.get("chosen", "").strip()
            rejected = record.get("rejected", "").strip()
            if prompt and chosen and rejected:
                self.samples.append(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(f"未从 {path} 读取到有效 DPO 样本")

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_pair(self, prompt_messages: List[dict], response: str) -> Dict[str, torch.Tensor]:
        messages = prompt_messages + [{"role": "assistant", "content": response}]
        text = format_messages(messages)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]
        labels = build_labels(
            input_ids, messages, tokenizer=self.tokenizer, max_length=self.max_length
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt_messages = dpo_prompt_to_messages(sample["prompt"])
        if not prompt_messages:
            prompt_messages = [{"role": "user", "content": sample["prompt"]}]

        chosen = self._encode_pair(prompt_messages, sample["chosen"])
        rejected = self._encode_pair(prompt_messages, sample["rejected"])

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
            "rejected_attention_mask": rejected["attention_mask"],
        }
