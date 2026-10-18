from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from harnessing_ts.state.workspace_store import WorkspaceStore


def _build_workspace(tmp_path) -> WorkspaceStore:
    store = WorkspaceStore(tmp_path)
    store.ensure_layout()
    return store


def test_write_main_llm_config_persists_and_merges(tmp_path) -> None:
    store = _build_workspace(tmp_path)

    written = store.write_main_llm_config(
        {
            "authMode": "manual",
            "protocol": "deepseek",
            "model": "deepseek-v4-pro",
            "apiKey": "sk-abcdef1234",
            "baseUrl": "https://api.deepseek.com/beta",
        }
    )
    assert written["model"] == "deepseek-v4-pro"
    assert written["apiKey"] == "sk-abcdef1234"

    raw = json.loads((tmp_path / "config.llm.json").read_text(encoding="utf-8"))
    assert raw["model"] == "deepseek-v4-pro"
    assert raw["apiKey"] == "sk-abcdef1234"
    assert raw["protocol"] == "deepseek"

    store.write_main_llm_config({"model": "qwen3.6-plus"})
    raw = json.loads((tmp_path / "config.llm.json").read_text(encoding="utf-8"))
    assert raw["model"] == "qwen3.6-plus"
    assert raw["apiKey"] == "sk-abcdef1234"


def test_write_main_llm_config_drops_empty_or_masked_secrets(tmp_path) -> None:
    store = _build_workspace(tmp_path)
    store.write_main_llm_config({"model": "m1", "apiKey": "first-secret"})

    store.write_main_llm_config({"apiKey": ""})
    raw = json.loads((tmp_path / "config.llm.json").read_text(encoding="utf-8"))
    assert raw["apiKey"] == "first-secret"

    store.write_main_llm_config({"apiKey": "****abcd"})
    raw = json.loads((tmp_path / "config.llm.json").read_text(encoding="utf-8"))
    assert raw["apiKey"] == "first-secret"


def test_write_main_llm_config_sanitizes_invalid_values(tmp_path) -> None:
    store = _build_workspace(tmp_path)
    written = store.write_main_llm_config(
        {
            "authMode": "injected-mode",
            "protocol": "fake-protocol",
            "model": 12345,  # type: ignore[dict-item]
        }
    )
    assert written["authMode"] == "sdk-default"
    assert written["protocol"] is None
    assert written["model"] == ""


def test_write_main_llm_config_round_trip_preserves_auth_mode(tmp_path) -> None:
    store = _build_workspace(tmp_path)
    assert store.read_main_llm_config()["authMode"] == "sdk-default"

    store.write_main_llm_config({"model": "first-model"})
    first = store.read_main_llm_config()

    store.write_main_llm_config({"model": "second-model"})
    second = store.read_main_llm_config()

    assert first["authMode"] == "sdk-default"
    assert second["authMode"] == "sdk-default"
    assert {key: value for key, value in second.items() if key != "model"} == {
        key: value for key, value in first.items() if key != "model"
    }
    assert second["model"] == "second-model"


def test_post_llm_config_endpoint_round_trips(tmp_path, monkeypatch) -> None:
    pytest.skip("FastAPI TestClient currently hangs in the project env; skipped temporarily.")

    from harnessing_ts.server import create_app

    _build_workspace(tmp_path)
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))

    client = TestClient(create_app())
    response = client.post(
        "/api/llm-config",
        json={
            "authMode": "manual",
            "protocol": "deepseek",
            "model": "endpoint-test-model",
            "apiKey": "sk-endpoint-key-1",
            "baseUrl": "https://example.com/v1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["config"]["model"] == "endpoint-test-model"
    assert payload["config"]["config"]["apiKey"].endswith("ey-1")
    assert (tmp_path / "config.llm.json").exists()

    raw = json.loads((tmp_path / "config.llm.json").read_text(encoding="utf-8"))
    assert raw["apiKey"] == "sk-endpoint-key-1"
