from __future__ import annotations

import pytest
from fastapi import HTTPException

from harnessing_ts.server import _static_file_target


def test_static_target_rejects_a_sibling_prefix(tmp_path) -> None:
    web_root = tmp_path / "dist"
    web_root.mkdir()
    sibling = tmp_path / "dist-secret"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        _static_file_target(web_root, "../dist-secret/secret.txt")

    assert exc.value.status_code == 403


def test_static_target_falls_back_to_index_for_unknown_paths(tmp_path) -> None:
    web_root = tmp_path / "dist"
    web_root.mkdir()
    (web_root / "index.html").write_text("ok", encoding="utf-8")

    assert _static_file_target(web_root, "missing/path") == (web_root / "index.html").resolve()
