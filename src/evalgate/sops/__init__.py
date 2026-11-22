"""SOP library loading and agent versions."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

LIBRARY_PATH = Path(__file__).parent / "library.yaml"


@dataclass(frozen=True)
class SOP:
    id: str
    title: str
    category: str
    priority: int
    text: str


def load_library(path: Path = LIBRARY_PATH) -> tuple[SOP, ...]:
    raw = yaml.safe_load(path.read_text())
    sops = tuple(
        SOP(**{k: (v.strip() if isinstance(v, str) else v) for k, v in s.items()}) for s in raw
    )
    ids = [s.id for s in sops]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SOP ids")
    return sops


@dataclass(frozen=True)
class AgentVersion:
    """Immutable snapshot of everything that defines agent behavior.

    The Agent Manager never edits a version; it creates a new one with a parent pointer.
    """

    sops: tuple[SOP, ...]
    model: str
    parent_id: str | None = None
    note: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.fingerprint())

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"model": self.model, "sops": [s.__dict__ for s in self.sops]}, sort_keys=True
        )
        return "v-" + hashlib.sha256(payload.encode()).hexdigest()[:10]

    def with_sop_text(self, sop_id: str, new_text: str, note: str = "") -> "AgentVersion":
        if sop_id not in {s.id for s in self.sops}:
            raise KeyError(sop_id)
        sops = tuple(
            SOP(**{**s.__dict__, "text": new_text}) if s.id == sop_id else s for s in self.sops
        )
        return AgentVersion(sops=sops, model=self.model, parent_id=self.id, note=note)

    def with_new_sop(self, sop: SOP, note: str = "") -> "AgentVersion":
        if sop.id in {s.id for s in self.sops}:
            raise ValueError(f"SOP {sop.id} already exists")
        return AgentVersion(sops=(*self.sops, sop), model=self.model, parent_id=self.id, note=note)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "model": self.model,
            "note": self.note,
            "sops": [s.__dict__ for s in self.sops],
        }

    @classmethod
    def from_json(cls, d: dict) -> "AgentVersion":
        v = cls(
            sops=tuple(SOP(**s) for s in d["sops"]),
            model=d["model"],
            parent_id=d.get("parent_id"),
            note=d.get("note", ""),
        )
        if d.get("id") and d["id"] != v.id:
            raise ValueError(f"version file says {d['id']} but its contents hash to {v.id}")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "AgentVersion":
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, directory: str | Path = "versions") -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.id}.json"
        p.write_text(json.dumps(self.to_json(), indent=1))
        return p
