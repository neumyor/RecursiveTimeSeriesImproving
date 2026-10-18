from __future__ import annotations

from harnessing_ts.mcp.server import _without_intend


def test_intend_is_removed_before_business_callback() -> None:
    assert _without_intend({"question": "Q", "intend": "说明调用意图"}) == {"question": "Q"}
