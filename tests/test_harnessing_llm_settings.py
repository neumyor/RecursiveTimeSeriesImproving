from __future__ import annotations

"""Settings-level LLM config tests: the maxTokens pass-through (the runtime's
256k default is rejected by gateways that validate against the model's own
output limit — the observed empty-turn failure mode)."""

import json
from pathlib import Path

from harnessing_ts.settings.llm import (
    build_sdk_invocation_config,
    mask_llm_config,
    read_effective_llm_config,
)


def _write_config(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "config.llm.json").write_text(json.dumps(payload), encoding="utf-8")


def test_max_tokens_passes_through_to_the_sdk_invocation(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "authMode": "manual",
            "model": "some-model",
            "apiKey": "sk-test",
            "baseUrl": "https://gateway.example/v1",
            "maxTokens": 16384,
        },
    )
    cfg = read_effective_llm_config(tmp_path)
    assert cfg.maxTokens == 16384
    sdk = build_sdk_invocation_config(cfg)
    assert sdk.max_tokens == 16384
    assert sdk.model == "some-model"
    assert mask_llm_config(cfg)["maxTokens"] == 16384


def test_invalid_max_tokens_is_dropped_not_rejected(tmp_path: Path) -> None:
    # The settings file is agent/user-editable; a bad value degrades to the
    # runtime default instead of breaking every session.
    for bad in ("bogus", -1, 0, True, 1.5):
        _write_config(tmp_path, {"authMode": "manual", "model": "m", "maxTokens": bad})
        assert read_effective_llm_config(tmp_path).maxTokens is None, bad


def test_env_var_overrides_the_file_for_max_tokens(tmp_path: Path, monkeypatch) -> None:
    _write_config(tmp_path, {"authMode": "manual", "model": "m", "maxTokens": 16384})
    monkeypatch.setenv("TS_HARNESS_LLM_MAX_TOKENS", "8192")
    assert read_effective_llm_config(tmp_path).maxTokens == 8192
    monkeypatch.setenv("TS_HARNESS_LLM_MAX_TOKENS", "not-a-number")
    assert read_effective_llm_config(tmp_path).maxTokens == 16384
