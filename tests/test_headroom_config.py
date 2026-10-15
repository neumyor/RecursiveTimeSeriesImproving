from __future__ import annotations

from pathlib import Path

import pytest

from harnessing_ts.headroom.config import HeadroomConfig


def test_defaults_are_valid() -> None:
    assert HeadroomConfig().validate() == []


def test_burn_in_combines_max_lag_and_fir_cap() -> None:
    cfg = HeadroomConfig()
    assert cfg.max_lag == 335
    assert cfg.burn_in == 335 + 256


def test_camel_case_round_trip(tmp_path: Path) -> None:
    cfg = HeadroomConfig(arm="bj", horizon=24, diag_a_ratio=0.6, seed=7)
    path = tmp_path / "config.headroom.json"
    cfg.to_json(path)
    assert HeadroomConfig.from_json(path) == cfg


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unexpectedKnob"):
        HeadroomConfig.from_dict({"unexpectedKnob": 1})


def test_validation_collects_all_errors() -> None:
    cfg = HeadroomConfig(input_length=0, arm="nope", horizon=-3, train_ratio=0.9, diag_ratio=0.1, cert_ratio=0.1)
    errors = cfg.validate()
    assert any("inputLength" in item for item in errors)
    assert any("arm" in item for item in errors)
    assert any("horizon" in item for item in errors)
    assert any("trainRatio" in item for item in errors)


def test_from_dict_rejects_invalid_config_with_all_reasons() -> None:
    with pytest.raises(ValueError, match="arm"):
        HeadroomConfig.from_dict({"arm": "nope"})
