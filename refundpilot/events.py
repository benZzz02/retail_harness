"""Append-only session logging and trace loading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class EventLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sequence = 0

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event


def session_event_path(runtime_dir: Path, session_id: str) -> Path:
    if Path(session_id).name != session_id or session_id in {"", ".", ".."}:
        raise ValueError("invalid session id")
    return runtime_dir / session_id / "events.jsonl"


def load_events(runtime_dir: Path, session_id: str) -> List[Dict[str, Any]]:
    path = session_event_path(runtime_dir, session_id)
    if not path.exists():
        raise FileNotFoundError("session trace not found: %s" % session_id)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

