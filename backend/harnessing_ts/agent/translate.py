from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any
from uuid import uuid4

from harnessing_ts.schema import Part


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def user_text_part(text: str) -> Part:
    return {"id": str(uuid4()), "timestamp": _now(), "role": "user", "type": "text", "text": text}


def system_text_part(text: str) -> Part:
    return {"id": str(uuid4()), "timestamp": _now(), "role": "system", "type": "text", "text": text}


def dsh_event_to_parts(
    event: Any,
    *,
    part_id: str | None = None,
    timestamp: str | None = None,
) -> list[Part]:
    """Project DSH JSON-RPC events into the provider-neutral transcript schema.

    DSH reports a call and its result as separate session events.  Keeping the
    callId as toolUseId lets MessageLog merge them into one semantic tool card.
    """
    if not isinstance(event, dict):
        return []
    event_type = event.get("type")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    base_id = part_id or str(uuid4())
    event_timestamp = timestamp or _now()

    if event_type == "assistant/message":
        message = data.get("message") if isinstance(data.get("message"), dict) else data
        content = message.get("content") if isinstance(message, dict) else None
        text = "".join(
            str(block.get("text") or "")
            for block in content or []
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            return []
        return [{
            "id": base_id,
            "timestamp": event_timestamp,
            "role": "assistant",
            "type": "text",
            "text": text,
            "raw": event,
        }]

    if event_type == "tool/call":
        name = data.get("name") if isinstance(data.get("name"), str) else "tool"
        input_value = _dsh_arguments(data.get("arguments"))
        call_id = data.get("callId")
        return [{
            "id": base_id,
            "timestamp": event_timestamp,
            "role": "assistant",
            "type": "tool_call",
            "name": name,
            "input": input_value,
            "intend": _extract_intend(input_value, name),
            "toolUseId": call_id if isinstance(call_id, str) else None,
            "status": "pending",
            "text": _extract_intend(input_value, name),
            "raw": event,
        }]

    if event_type == "tool/result":
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        source = message.get("source") if isinstance(message.get("source"), dict) else {}
        source_call_id = source.get("callId") if isinstance(source.get("callId"), str) else None
        content = message.get("content") if isinstance(message.get("content"), list) else []
        parts: list[Part] = []
        for index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool-result":
                continue
            call_id = block.get("toolCallId") if isinstance(block.get("toolCallId"), str) else source_call_id
            failed = block.get("isError") is True
            result_text = _format_tool_result(block.get("content"))
            parts.append({
                "id": base_id if index == 0 else f"{base_id}:{index}",
                "timestamp": event_timestamp,
                "role": "tool",
                "type": "tool_result",
                "toolUseId": call_id,
                "status": "failed" if failed else "completed",
                "resultText": result_text,
                "text": result_text,
                "raw": event,
            })
        return parts

    return []


def merge_tool_result_part(tool_call: Part, tool_result: Part) -> Part:
    merged = dict(tool_call)
    merged["status"] = "failed" if tool_result.get("status") == "failed" else "completed"
    merged["resultText"] = _strip_reasoning_jsonl_lines(tool_result.get("resultText") or tool_result.get("text") or "")
    merged["resultRaw"] = tool_result.get("raw")
    merged["text"] = merged.get("intend") or _format_tool_use(merged.get("name"), merged.get("input"))
    return merged


def should_merge_tool_result(tool_call: Part, tool_result: Part) -> bool:
    if tool_call.get("type") != "tool_call" or tool_result.get("type") != "tool_result":
        return False
    call_id = tool_call.get("toolUseId")
    result_id = tool_result.get("toolUseId")
    return isinstance(call_id, str) and call_id != "" and call_id == result_id


def collapse_tool_parts(parts: list[Part]) -> list[Part]:
    out: list[Part] = []
    for part in parts:
        if part.get("type") == "tool_result":
            merged = False
            for index in range(len(out) - 1, -1, -1):
                if should_merge_tool_result(out[index], part):
                    out[index] = merge_tool_result_part(out[index], part)
                    merged = True
                    break
            if merged:
                continue
        out.append(part)
    return out


def _dsh_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"rawArguments": value}
        return parsed
    return {}


def filter_display_parts(parts: list[Part]) -> list[Part]:
    """Return the bounded public projection of an audit transcript.

    The JSONL transcript intentionally retains the provider event so an
    operator can audit a run.  That object can contain full shell output,
    tool arguments, and provider-only metadata, and must not be embedded in
    a bootstrap/SSE payload merely because a UI happens not to render it.
    """
    return [public_display_part(_sanitize_display_part(part)) for part in parts]


_PUBLIC_PART_KEYS = {
    "id", "timestamp", "role", "type", "text", "name", "input", "intend",
    "toolUseId", "status", "resultText", "displayText", "sourceLabel",
}
_PUBLIC_TEXT_LIMIT = 6_000
_PUBLIC_INPUT_LIMIT = 2_000


def public_display_part(part: Part) -> Part:
    """Remove audit-only payloads and cap display strings before transport."""
    public = {key: value for key, value in part.items() if key in _PUBLIC_PART_KEYS}
    for key in ("text", "intend", "resultText", "displayText"):
        value = public.get(key)
        if isinstance(value, str):
            public[key] = _truncate_display_text(value, _PUBLIC_TEXT_LIMIT)
    if "input" in public:
        public["input"] = _bounded_display_value(public["input"], _PUBLIC_INPUT_LIMIT)
    return public


def _truncate_display_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n… [已截断；完整审计记录保存在 workspace JSONL 日志中]"


def _bounded_display_value(value: Any, limit: int) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return _truncate_display_text(str(value), limit)
    if len(encoded) <= limit:
        return value
    return {"displaySummary": _truncate_display_text(encoded, limit), "truncated": True}


def _sanitize_display_part(part: Part) -> Part:
    if part.get("type") not in {"tool_call", "tool_result"}:
        return part
    changed = False
    sanitized = dict(part)
    for key in ("resultText", "text"):
        value = sanitized.get(key)
        if isinstance(value, str):
            clean = _strip_reasoning_jsonl_lines(value)
            if clean != value:
                sanitized[key] = clean
                changed = True
    return sanitized if changed else part


def _raw_contains_only_reasoning_content(raw: Any) -> bool:
    content = _content(raw) if isinstance(raw, dict) else []
    if not content:
        return False
    for item in content:
        if isinstance(item, str) and item.strip():
            return False
        if not isinstance(item, dict):
            return False
        if isinstance(item.get("text"), str) and item["text"].strip():
            return False
        if _is_tool_use(item) or _is_tool_result(item):
            return False
        if item.get("type") not in {None, "thinking", "redacted_thinking"} and not any(
            key in item for key in ("thinking", "signature", "redacted_thinking")
        ):
            return False
        if not any(key in item for key in ("thinking", "signature", "redacted_thinking")) and item.get("type") not in {"thinking", "redacted_thinking"}:
            return False
    return True


def _content(raw: dict[str, Any]) -> list[Any]:
    message = raw.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        return message["content"]
    if isinstance(raw.get("content"), list):
        return raw["content"]
    return []


def _is_tool_use(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and "input" in item
        and (item.get("type") in {None, "tool_use"} or isinstance(item.get("id"), str))
    )


def _is_tool_result(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "tool_result" and "content" in item


def _format_tool_use(name: Any, input_value: Any) -> str:
    label = name if isinstance(name, str) else "tool"
    if input_value is None:
        return f"Tool call: {label}"
    return f"Tool call: {label}\n{json.dumps(input_value, ensure_ascii=False, indent=2)}"


def _extract_intend(input_value: Any, name: Any) -> str:
    if isinstance(input_value, dict):
        value = input_value.get("intend")
        if isinstance(value, str) and value.strip():
            return value.strip()
        description = input_value.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    label = name if isinstance(name, str) else "tool"
    return f"调用工具：{label}"


def _format_tool_result(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return _strip_reasoning_jsonl_lines(content)
    extracted = _extract_tool_result_text(content)
    if extracted:
        return _strip_reasoning_jsonl_lines(extracted)
    return "Tool result"


def _strip_reasoning_jsonl_lines(text: Any) -> str:
    if not isinstance(text, str) or not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return text
    kept: list[str] = []
    for line in lines:
        if _is_ignorable_reasoning_json_line(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _is_ignorable_reasoning_json_line(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    subtype = parsed.get("subtype")
    if isinstance(subtype, str) and (subtype == "thinking_tokens" or subtype.endswith("_tokens")):
        return True
    return _raw_contains_only_reasoning_content(parsed)


def _extract_tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            continue
        nested = _extract_tool_result_text(item.get("content"))
        if nested:
            parts.append(nested)
    return "\n".join(parts)
