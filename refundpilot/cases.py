"""Load the visible sample workload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .contracts import TaskCase


def default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "cases.json"


def load_cases(path: Optional[Path] = None) -> List[TaskCase]:
    source = path or default_cases_path()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("cases file must contain a JSON list")
    cases = [TaskCase.from_mapping(item) for item in raw]
    task_ids = [case.task_id for case in cases]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique")
    return cases


def find_case(task_id: str, path: Optional[Path] = None) -> TaskCase:
    for case in load_cases(path):
        if case.task_id == task_id:
            return case
    raise KeyError("unknown case: %s" % task_id)

