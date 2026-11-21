"""Append-only conversation traces. Every session is replayable and re-scorable from its trace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trace:
    session_id: str
    agent_version: str
    model: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, type_: str, **data: Any) -> None:
        self.events.append({"t": round(time.time(), 4), "type": type_, **data})

    def of_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type_]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_version": self.agent_version,
            "model": self.model,
            "events": self.events,
        }

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.session_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        return path
