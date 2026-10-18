from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

T = TypeVar("T")


def _write_text_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step.

    The content is written to a temp sibling and moved into place with
    ``os.replace``, so a crash mid-write can never leave a truncated
    JSON file behind; readers see either the old or the new content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, values: list[Any]) -> None:
    text = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
    _write_text_atomically(path, text)


def read_jsonl(path: Path) -> list[Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out: list[Any] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_json(path: Path, value: Any) -> None:
    _write_text_atomically(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def clear_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
