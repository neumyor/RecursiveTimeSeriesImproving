"""Token-usage accounting over append-only DSH session logs (zstd JSONL)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

try:
    import zstandard
except ImportError:  # pragma: no cover - only hit without the research extras
    zstandard = None  # type: ignore[assignment]

# Source field names are kept verbatim (camelCase) so the sums can be diffed
# against the raw logs without a mapping table.
_FIELDS = ("inputTokens", "outputTokens", "cacheReadTokens", "reasoningTokens")

_INSTALL_HINT = (
    "zstandard is required to read DSH session logs; "
    "install the research extras: uv run --extra research ... (or uv sync --extra research)"
)


def _zero_usage() -> dict[str, int]:
    """Fresh accumulator: every field defaults to 0 so missing data never NaNs."""
    totals: dict[str, int] = {field: 0 for field in _FIELDS}
    totals["steps"] = 0
    totals["skippedLines"] = 0
    return totals


def _usage_dict(obj: object) -> dict[str, Any] | None:
    """First dict carrying token counts, data level before top level.

    DSH writes ``data.usage`` on assistant/message events; the fallbacks
    tolerate producers that hoist the same dict elsewhere rather than
    silently undercounting a whole session.  Presence of ``inputTokens``
    or ``outputTokens`` is the marker — the fields are optional per line.
    """
    if not isinstance(obj, dict):
        return None
    data = obj.get("data")
    candidates: tuple[object, ...] = (
        data.get("usage") if isinstance(data, dict) else None,
        data,
        obj.get("usage"),
        obj,
    )
    for cand in candidates:
        if isinstance(cand, dict) and ("inputTokens" in cand or "outputTokens" in cand):
            return cand
    return None


def session_usage(path: Path) -> dict[str, int]:
    """Token totals for one ``session.jsonl.zstd``.

    ``steps`` counts lines carrying a usage dict rather than every
    assistant event: a usage-bearing line is one billable model call, and
    counting events without usage would dilute per-step averages.  Lines
    that fail json.loads are tallied in ``skippedLines`` instead of
    aborting the sum — one torn line must not hide the rest of a session.
    """
    if zstandard is None:
        raise RuntimeError(_INSTALL_HINT)
    totals = _zero_usage()
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        # BufferedReader supplies line iteration on top of the streaming
        # decompressor, so a large session never sits fully in memory.
        with io.BufferedReader(dctx.stream_reader(fh)) as reader:
            for raw in reader:
                try:
                    obj = json.loads(raw)
                except ValueError:  # JSONDecodeError and UnicodeDecodeError alike
                    totals["skippedLines"] += 1
                    continue
                usage = _usage_dict(obj)
                if usage is None:
                    continue
                totals["steps"] += 1
                for field in _FIELDS:
                    value = usage.get(field, 0)
                    # bool is an int subclass but true/false is corruption, not a count
                    if not isinstance(value, int) or isinstance(value, bool):
                        raise ValueError(
                            f"{path}: usage field {field!r} must be an int, got {value!r}"
                        )
                    totals[field] += value
    return totals


def workspace_usage(workspace: Path) -> dict[str, int]:
    """Sum :func:`session_usage` over ``workspace/.dsh-sessions/**/session.jsonl.zstd``.

    A missing or empty sessions directory is a fresh workspace, not an
    error: zeros with ``sessions=0`` let callers render "no runs yet".
    """
    totals = _zero_usage()
    totals["sessions"] = 0
    sessions_dir = workspace / ".dsh-sessions"
    if not sessions_dir.is_dir():
        return totals
    for log in sessions_dir.rglob("session.jsonl.zstd"):
        session = session_usage(log)
        totals["sessions"] += 1
        for key, value in session.items():
            totals[key] += value
    return totals
