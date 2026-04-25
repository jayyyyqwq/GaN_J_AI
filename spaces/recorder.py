"""
Serialize/deserialize list[Snapshot] to/from JSON for pre-recorded episodes.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .runner import Snapshot

_PRERECORDED_DIR = Path(__file__).parent / "prerecorded"


def save_episode(history: list[Snapshot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [dataclasses.asdict(s) for s in history]
    path.write_text(json.dumps(data, indent=2))


def load_episode(path: Path) -> list[Snapshot]:
    data = json.loads(path.read_text())
    return [Snapshot(**d) for d in data]


def load_prerecorded(name: str) -> list[Snapshot] | None:
    """Load a named pre-recorded episode. Returns None if file not found."""
    path = _PRERECORDED_DIR / f"{name}.json"
    if not path.exists():
        return None
    return load_episode(path)
