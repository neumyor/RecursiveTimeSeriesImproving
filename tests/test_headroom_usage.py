from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from harnessing_ts.headroom.usage import session_usage, workspace_usage


def _write_session(path: Path, lines: list[str | dict]) -> Path:
    """Compress ``lines`` (objects JSON-encoded, strings verbatim) as a session log."""
    payload = "\n".join(
        line if isinstance(line, str) else json.dumps(line) for line in lines
    )
    payload += "\n"
    cctx = zstandard.ZstdCompressor()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(cctx.compress(payload.encode()))
    return path


def _assistant(usage: dict) -> dict:
    """One assistant/message event in the on-disk DSH shape."""
    return {"type": "assistant/message", "data": {"usage": usage}}


def test_session_usage_sums_usage_lines_and_skips_malformed(tmp_path: Path) -> None:
    log = _write_session(
        tmp_path / "session.jsonl.zstd",
        [
            {"type": "session", "cwd": "/tmp"},  # header: no usage
            _assistant(
                {
                    "inputTokens": 100,
                    "outputTokens": 10,
                    "cacheReadTokens": 5,
                    "reasoningTokens": 2,
                }
            ),
            {"type": "user/message", "data": {"content": "hi"}},  # no usage
            _assistant({"inputTokens": 50, "outputTokens": 5}),  # missing fields
            "{not valid json",  # torn line
            {"usage": {"inputTokens": 7, "outputTokens": 3}},  # top-level tolerance
        ],
    )
    totals = session_usage(log)
    assert totals == {
        "inputTokens": 157,
        "outputTokens": 18,
        "cacheReadTokens": 5,
        "reasoningTokens": 2,
        "steps": 3,
        "skippedLines": 1,
    }


def test_workspace_usage_sums_nested_sessions(tmp_path: Path) -> None:
    _write_session(
        tmp_path / ".dsh-sessions" / "alpha" / "session.jsonl.zstd",
        [_assistant({"inputTokens": 10, "outputTokens": 1})],
    )
    _write_session(
        tmp_path / ".dsh-sessions" / "nested" / "beta" / "session.jsonl.zstd",
        [_assistant({"inputTokens": 20, "cacheReadTokens": 4})],
    )
    # Non-session files must not be picked up by the recursive scan.
    (tmp_path / ".dsh-sessions" / "notes.txt").write_text("ignore me")
    totals = workspace_usage(tmp_path)
    assert totals["sessions"] == 2
    assert totals["inputTokens"] == 30
    assert totals["outputTokens"] == 1
    assert totals["cacheReadTokens"] == 4
    assert totals["reasoningTokens"] == 0
    assert totals["steps"] == 2
    assert totals["skippedLines"] == 0


def test_missing_file_and_empty_dir_yield_zeros(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        session_usage(tmp_path / "nope" / "session.jsonl.zstd")

    empty = workspace_usage(tmp_path)  # no .dsh-sessions at all
    assert empty == {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadTokens": 0,
        "reasoningTokens": 0,
        "steps": 0,
        "skippedLines": 0,
        "sessions": 0,
    }

    (tmp_path / ".dsh-sessions").mkdir()
    assert workspace_usage(tmp_path)["sessions"] == 0


def test_missing_usage_fields_count_as_zero(tmp_path: Path) -> None:
    log = _write_session(
        tmp_path / "session.jsonl.zstd",
        [_assistant({"inputTokens": 42})],
    )
    totals = session_usage(log)
    assert totals["inputTokens"] == 42
    assert totals["outputTokens"] == 0
    assert totals["cacheReadTokens"] == 0
    assert totals["reasoningTokens"] == 0
    assert totals["steps"] == 1


def test_non_int_usage_field_raises(tmp_path: Path) -> None:
    # A float or string count means the log is corrupt; silently coercing it
    # would fabricate numbers, so the contract demands ValueError.
    log = _write_session(
        tmp_path / "session.jsonl.zstd",
        [_assistant({"inputTokens": "many"})],
    )
    with pytest.raises(ValueError, match="inputTokens"):
        session_usage(log)
