"""Machine-readable trace events and append-only JSONL persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    timestamp: str
    component: str
    event_type: str
    status: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlTraceWriter:
    """Write one UTF-8 JSON object per line without buffering whole runs."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def path_for(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.jsonl"

    def append(self, event: TraceEvent) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.path_for(event.run_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
