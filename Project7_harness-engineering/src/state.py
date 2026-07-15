"""使用普通文件保存 Harness 运行状态；便于检查、恢复和回放。"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, runs_dir: str | Path):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        # run_id 由本类生成；这里仍禁止调用者通过路径片段访问 runs_dir 外部。
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("非法 run_id")
        return self.runs_dir / run_id

    def create(self, task: str, config: dict) -> str:
        run_id = uuid.uuid4().hex[:16]
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id,
            "task": task,
            "config": config,
            "created_at": _now(),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "trace.jsonl").touch()
        return run_id

    def append(self, run_id: str, event: dict) -> None:
        record = dict(event)
        record.setdefault("timestamp", _now())
        trace_path = self._run_dir(run_id) / "trace.jsonl"
        with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()

    def load(self, run_id: str) -> dict:
        run_dir = self._run_dir(run_id)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {**metadata, "events": events}

    def trace_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "trace.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id)

