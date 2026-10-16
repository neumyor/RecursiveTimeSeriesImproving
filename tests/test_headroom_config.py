from __future__ import annotations

from pathlib import Path

import pytest

from harnessing_ts.headroom.config import ARM_VALUES, HeadroomConfig


def test_defaults_are_valid() -> None:
    assert HeadroomConfig().validate() == []


def test_burn_in_combines_max_lag_and_fir_cap() -> None:
    cfg = HeadroomConfig()
    assert cfg.max_lag == 335
    assert cfg.burn_in == 335 + 256


def test_camel_case_round_trip(tmp_path: Path) -> None:
    cfg = HeadroomConfig(horizon=24, diag_a_ratio=0.6, seed=7, evolution=False)
    path = tmp_path / "config.headroom.json"
    cfg.to_json(path)
    assert HeadroomConfig.from_json(path) == cfg


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unexpectedKnob"):
        HeadroomConfig.from_dict({"unexpectedKnob": 1})


def test_feature_depth_beyond_burn_in_is_rejected() -> None:
    # beyond_l_lags so large its feature reach exceeds the burn-in (the
    # wrapping-to-future-rows bug validate() exists to stop).
    cfg = HeadroomConfig(beyond_l_lags=64)
    assert any("beyond-l" in item for item in cfg.validate())


def test_overlapping_lag_segments_are_rejected() -> None:
    cfg = HeadroomConfig(lag_segments=((0, 23), (23, 95)))
    assert any("lagSegments" in item for item in cfg.validate())


def test_split_ratios_leaving_no_test_are_rejected() -> None:
    cfg = HeadroomConfig(train_ratio=0.9, diag_ratio=0.1, cert_ratio=0.1)
    assert any("trainRatio" in item for item in cfg.validate())


def test_from_dict_rejects_invalid_config_with_all_reasons() -> None:
    with pytest.raises(ValueError, match="lagSegments"):
        HeadroomConfig.from_dict({"lagSegments": [[5, 3]]})


def test_arm_values_match_the_loop_choices() -> None:
    assert ARM_VALUES == ("scalar", "fixed", "self-built", "augmented", "placebo")
