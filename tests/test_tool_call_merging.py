from __future__ import annotations

import os
from pathlib import Path

import pytest

from harnessing_ts.agent.translate import (
    collapse_tool_parts,
    dsh_event_to_parts,
    filter_display_parts,
)
from harnessing_ts.state.jsonl import read_jsonl
from harnessing_ts.state.message_log import MessageLog


def _dsh_tool_call(*, name: str = "mcp__ts_harness__scan_references", arguments: str | None = None) -> dict:
    return {
        "type": "tool/call",
        "seq": 101,
        "data": {
            "callId": "call_dsh_1",
            "name": name,
            "arguments": arguments or '{"intend":"扫描参考文献以发现新增资料","include_processed":true}',
        },
    }


def _dsh_tool_result(*, failed: bool = False, text: str | None = None) -> dict:
    result_text = text if text is not None else ("权限不足" if failed else "扫描完成")
    return {
        "type": "tool/result",
        "seq": 102,
        "data": {
            "message": {
                "source": {"kind": "tool", "callId": "call_dsh_1"},
                "content": [{
                    "type": "tool-result",
                    "toolCallId": "call_dsh_1",
                    "content": [{"type": "text", "text": result_text}],
                    "isError": failed,
                }],
            },
        },
    }


def test_tool_call_and_result_collapse_to_one_tool_call_part() -> None:
    collapsed = collapse_tool_parts([
        dsh_event_to_parts(_dsh_tool_call())[0],
        dsh_event_to_parts(_dsh_tool_result())[0],
    ])

    assert len(collapsed) == 1
    assert collapsed[0]["type"] == "tool_call"
    assert collapsed[0]["status"] == "completed"
    assert collapsed[0]["intend"] == "扫描参考文献以发现新增资料"
    assert collapsed[0]["resultText"] == "扫描完成"


def test_tool_result_strips_reasoning_only_jsonl_lines() -> None:
    result_text = "\n".join([
        "before",
        '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"private","signature":"sig"}]}}',
        '{"type":"system","subtype":"thinking_tokens","estimated_tokens":42}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"visible jsonl text"}]}}',
        "after",
    ])

    collapsed = collapse_tool_parts([
        dsh_event_to_parts(_dsh_tool_call())[0],
        dsh_event_to_parts(_dsh_tool_result(text=result_text))[0],
    ])

    result = collapsed[0]["resultText"]
    assert "before" in result
    assert "after" in result
    assert "visible jsonl text" in result
    assert "private" not in result
    assert "thinking_tokens" not in result


def test_display_filter_sanitizes_merged_tool_call_result_text() -> None:
    tool_call = {
        "role": "assistant",
        "type": "tool_call",
        "name": "bash",
        "text": "inspect subagent logs",
        "resultText": "\n".join([
            "before",
            '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"private","signature":"sig"}]}}',
            "after",
        ]),
    }

    parts = filter_display_parts([tool_call])

    assert parts[0]["resultText"] == "before\nafter"


def test_message_log_merges_tool_result_into_prior_tool_call(tmp_path) -> None:
    log = MessageLog(tmp_path / "main.jsonl")
    log.append(dsh_event_to_parts(_dsh_tool_call())[0])
    log.append(dsh_event_to_parts(_dsh_tool_result())[0])

    parts = read_jsonl(log.path)

    assert len(parts) == 1
    assert parts[0]["type"] == "tool_call"
    assert parts[0]["status"] == "completed"
    assert parts[0]["resultText"] == "扫描完成"


def test_dsh_tool_events_become_intent_bearing_correlated_tool_card() -> None:
    tool_call = dsh_event_to_parts(_dsh_tool_call())[0]
    tool_result = dsh_event_to_parts(_dsh_tool_result())[0]

    collapsed = collapse_tool_parts([tool_call, tool_result])

    assert len(collapsed) == 1
    assert collapsed[0]["type"] == "tool_call"
    assert collapsed[0]["name"] == "mcp__ts_harness__scan_references"
    assert collapsed[0]["intend"] == "扫描参考文献以发现新增资料"
    assert collapsed[0]["toolUseId"] == "call_dsh_1"
    assert collapsed[0]["status"] == "completed"
    assert collapsed[0]["resultText"] == "扫描完成"


def test_dsh_native_tool_uses_description_as_intend() -> None:
    tool_call = dsh_event_to_parts(_dsh_tool_call(
        name="bash",
        arguments='{"command":"ls -la","description":"列出工作区文件以了解现状"}',
    ))[0]

    assert tool_call["intend"] == "列出工作区文件以了解现状"
    assert tool_call["input"]["command"] == "ls -la"


def test_dsh_tool_failure_is_merged_into_the_original_card() -> None:
    collapsed = collapse_tool_parts([
        dsh_event_to_parts(_dsh_tool_call())[0],
        dsh_event_to_parts(_dsh_tool_result(failed=True))[0],
    ])

    assert collapsed[0]["status"] == "failed"
    assert collapsed[0]["resultText"] == "权限不足"


def test_display_filter_drops_audit_payload_and_bounds_large_tool_output() -> None:
    parts = filter_display_parts([{
        "id": "tool-1",
        "role": "assistant",
        "type": "tool_call",
        "name": "bash",
        "input": {"command": "x" * 5000},
        "intend": "检查工作区",
        "resultText": "y" * 7000,
        "raw": {"secret": "must remain audit-only"},
        "resultRaw": {"large": "must remain audit-only"},
    }])

    assert "raw" not in parts[0]
    assert "resultRaw" not in parts[0]
    assert parts[0]["input"]["truncated"] is True
    assert "已截断" in parts[0]["resultText"]


def test_write_json_and_jsonl_go_through_a_temp_sibling_and_replace(tmp_path, monkeypatch) -> None:
    from harnessing_ts.state import jsonl

    seen: list[tuple[object, object]] = []
    real_replace = os.replace

    def tracked_replace(src, dst) -> None:
        seen.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(jsonl.os, "replace", tracked_replace)

    json_path = tmp_path / "state" / "execution.json"
    jsonl.write_json(json_path, {"phase": "running"})
    assert len(seen) == 1
    assert seen[0][1] == json_path
    # The replacement source is a distinct temp sibling, never the target.
    assert Path(seen[0][0]).parent == json_path.parent
    assert Path(seen[0][0]) != json_path

    jsonl_path = tmp_path / "logs" / "main.jsonl"
    jsonl.write_jsonl(jsonl_path, [{"id": "a"}, {"id": "b"}])
    assert len(seen) == 2
    assert Path(seen[1][0]).parent == jsonl_path.parent

    assert jsonl.read_json(json_path) == {"phase": "running"}
    assert jsonl.read_jsonl(jsonl_path) == [{"id": "a"}, {"id": "b"}]
    # No temp files are left behind next to the written files.
    assert [p.name for p in (tmp_path / "state").iterdir()] == ["execution.json"]
    assert [p.name for p in (tmp_path / "logs").iterdir()] == ["main.jsonl"]


def test_failed_replace_leaves_the_original_file_untouched(tmp_path, monkeypatch) -> None:
    from harnessing_ts.state import jsonl

    json_path = tmp_path / "state" / "execution.json"
    jsonl.write_json(json_path, {"phase": "ready"})
    original = json_path.read_text(encoding="utf-8")

    def failing_replace(_src, _dst) -> None:
        raise OSError("crashed before the swap")

    monkeypatch.setattr(jsonl.os, "replace", failing_replace)
    with pytest.raises(OSError, match="crashed before the swap"):
        jsonl.write_json(json_path, {"phase": "running"})

    assert json_path.read_text(encoding="utf-8") == original
    assert [p.name for p in (tmp_path / "state").iterdir()] == ["execution.json"]


def test_message_log_tool_result_merge_keeps_original_on_failed_replace(tmp_path, monkeypatch) -> None:
    from harnessing_ts.state import jsonl

    log = MessageLog(tmp_path / "main.jsonl")
    log.append(dsh_event_to_parts(_dsh_tool_call())[0])
    original = log.path.read_text(encoding="utf-8")

    def failing_replace(_src, _dst) -> None:
        raise OSError("crashed before the swap")

    monkeypatch.setattr(jsonl.os, "replace", failing_replace)
    with pytest.raises(OSError, match="crashed before the swap"):
        log.append(dsh_event_to_parts(_dsh_tool_result())[0])

    # The audit log still holds the unmerged tool call, byte for byte.
    assert log.path.read_text(encoding="utf-8") == original
    assert [part["type"] for part in read_jsonl(log.path)] == ["tool_call"]
