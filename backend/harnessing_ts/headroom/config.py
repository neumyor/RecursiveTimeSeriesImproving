from __future__ import annotations

"""Frozen experiment configuration for the residual-headroom system.

The camelCase JSON mapping and the validation rules are binding; see
docs/headroom-contracts.md §2.
"""

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


ARM_VALUES = ("scalar", "bj", "card", "placebo")

_FIELD_ALIASES: dict[str, str] = {
    "input_length": "inputLength",
    "horizon": "horizon",
    "arm": "arm",
    "budget_candidates": "budgetCandidates",
    "candidate_timeout_seconds": "candidateTimeoutSeconds",
    "acceptance_alpha": "acceptanceAlpha",
    "certify_alpha": "certifyAlpha",
    "certify_epsilon": "certifyEpsilon",
    "lag_segments": "lagSegments",
    "horizon_realized_lags": "horizonRealizedLags",
    "max_period_peaks": "maxPeriodPeaks",
    "band_period_range": "bandPeriodRange",
    "fir_len_cap": "firLenCap",
    "beyond_l_lags": "beyondLLags",
    "channel_max": "channelMax",
    "ridge_alphas": "ridgeAlphas",
    "train_ratio": "trainRatio",
    "diag_ratio": "diagRatio",
    "cert_ratio": "certRatio",
    "diag_a_ratio": "diagARatio",
    "seed": "seed",
    "bootstrap_runs": "bootstrapRuns",
}
_JSON_ALIASES: dict[str, str] = {value: key for key, value in _FIELD_ALIASES.items()}


@dataclass(frozen=True)
class HeadroomConfig:
    """All knobs of one experiment run; see headroom-contracts.md §2."""

    input_length: int = 96
    horizon: int = 24
    arm: str = "card"
    budget_candidates: int = 25
    candidate_timeout_seconds: float = 600.0
    acceptance_alpha: float = 0.05
    certify_alpha: float = 0.05
    certify_epsilon: float = 0.02
    lag_segments: tuple[tuple[int, int], ...] = ((0, 23), (24, 95), (96, 335))
    horizon_realized_lags: int = 4
    max_period_peaks: int = 3
    band_period_range: tuple[int, int] = (4, 400)
    fir_len_cap: int = 256
    beyond_l_lags: int = 10
    channel_max: int = 8
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    train_ratio: float = 0.6
    diag_ratio: float = 0.2
    cert_ratio: float = 0.1
    diag_a_ratio: float = 0.5
    seed: int = 0
    bootstrap_runs: int = 200

    @property
    def max_lag(self) -> int:
        return self.lag_segments[-1][1]

    @property
    def burn_in(self) -> int:
        """Origins before this index cannot have fully realized features."""
        return self.max_lag + self.fir_len_cap

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.input_length <= 0:
            errors.append("inputLength must be positive")
        if self.lag_segments and self.input_length > 0 and self.burn_in < self.input_length:
            # Standalone cert/test arrays index windows from the segment start,
            # so the burn-in must cover at least the full lookback.
            errors.append(
                f"burnIn (maxLag + firLenCap = {self.burn_in}) must be >= inputLength ({self.input_length})"
            )
        if self.horizon <= 0:
            errors.append("horizon must be positive")
        if self.arm not in ARM_VALUES:
            errors.append(f"arm must be one of {ARM_VALUES}")
        if not 1 <= self.budget_candidates <= 500:
            errors.append("budgetCandidates must be in [1, 500]")
        if self.candidate_timeout_seconds <= 0:
            errors.append("candidateTimeoutSeconds must be positive")
        for name, value in (("acceptanceAlpha", self.acceptance_alpha), ("certifyAlpha", self.certify_alpha)):
            if not 0.0 < value < 1.0:
                errors.append(f"{name} must be in (0, 1)")
        if self.certify_epsilon <= 0:
            errors.append("certifyEpsilon must be positive")
        if not self.lag_segments:
            errors.append("lagSegments must not be empty")
        else:
            previous_end = -1
            for k1, k2 in self.lag_segments:
                if not 0 <= k1 <= k2:
                    errors.append(f"lagSegments entries must satisfy 0 <= k1 <= k2, got ({k1}, {k2})")
                if k1 <= previous_end:
                    errors.append("lagSegments must be sorted and non-overlapping")
                previous_end = k2
        if not 1 <= self.horizon_realized_lags <= 16:
            errors.append("horizonRealizedLags must be in [1, 16]")
        if not 1 <= self.max_period_peaks <= 10:
            errors.append("maxPeriodPeaks must be in [1, 10]")
        lo, hi = self.band_period_range
        if not 2 <= lo < hi <= 5000:
            errors.append("bandPeriodRange must satisfy 2 <= lo < hi <= 5000")
        if self.fir_len_cap < 16:
            errors.append("firLenCap must be >= 16")
        if not 1 <= self.beyond_l_lags <= 64:
            errors.append("beyondLLags must be in [1, 64]")
        if not 1 <= self.channel_max <= 64:
            errors.append("channelMax must be in [1, 64]")
        if not self.ridge_alphas or any(alpha <= 0 for alpha in self.ridge_alphas):
            errors.append("ridgeAlphas must be a non-empty list of positive values")
        for name, value in (("trainRatio", self.train_ratio), ("diagRatio", self.diag_ratio), ("certRatio", self.cert_ratio)):
            if not 0.0 < value < 1.0:
                errors.append(f"{name} must be in (0, 1)")
        if self.train_ratio + self.diag_ratio + self.cert_ratio >= 1.0:
            errors.append("trainRatio + diagRatio + certRatio must be < 1 (the rest is test)")
        if not 0.0 < self.diag_a_ratio < 1.0:
            errors.append("diagARatio must be in (0, 1)")
        if self.seed < 0:
            errors.append("seed must be >= 0")
        if not 0 <= self.bootstrap_runs <= 10000:
            errors.append("bootstrapRuns must be in [0, 10000]")
        return errors

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "lag_segments":
                value = [[k1, k2] for k1, k2 in value]
            elif field.name == "band_period_range":
                value = list(value)
            elif field.name == "ridge_alphas":
                value = list(value)
            raw[_FIELD_ALIASES[field.name]] = value
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HeadroomConfig:
        known = set(_JSON_ALIASES)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown HeadroomConfig keys: {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for key, value in raw.items():
            name = _JSON_ALIASES[key]
            if name == "lag_segments":
                value = tuple((int(k1), int(k2)) for k1, k2 in value)
            elif name in {"band_period_range", "ridge_alphas"}:
                value = tuple(value)
            values[name] = value
        config = cls(**values)
        errors = config.validate()
        if errors:
            raise ValueError("Invalid HeadroomConfig:\n" + "\n".join(f"- {item}" for item in errors))
        return config

    @classmethod
    def from_json(cls, path: Path) -> HeadroomConfig:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"HeadroomConfig JSON is malformed: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("HeadroomConfig JSON must be an object")
        return cls.from_dict(raw)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
