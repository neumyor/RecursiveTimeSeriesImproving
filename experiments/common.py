"""Shared helpers for the experiments/ drivers (loaded by file path, not a package).

Every driver is executed standalone (``python experiments/<driver>.py``) or
loaded by tests via importlib, so this module is imported after the driver
puts experiments/ on sys.path — it must never import sibling drivers itself.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def load_module(name: str, source: Path) -> Any:
    """Import a file that is not an installed package and return the module.

    Cached: an already-registered name returns the existing module, so
    loading the same canonical name from several drivers shares one object.
    Registered in sys.modules before exec so dataclasses inside the loaded
    source can resolve string annotations.
    """
    if name in sys.modules:
        return sys.modules[name]
    return exec_module(name, source)


def exec_module(name: str, source: Path) -> Any:
    """Like load_module but always re-executes the file.

    For sources whose file content changes between loads (the workspace
    incumbent, a fresh candidate): a cached lookup would hand back the
    previous model's code.
    """
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(value: Any) -> Any:
    """Non-finite floats -> None recursively; numpy scalars -> plain Python.

    Infinities are dropped too: a divergent candidate can leave an inf MSE in
    a report, and json.dumps with allow_nan=False would refuse to write it.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant read: a torn LAST line (kill mid-append) is dropped."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                return records  # torn tail from an interrupted append
            raise ValueError(f"{path} is corrupted at line {index + 1}") from None
    return records
