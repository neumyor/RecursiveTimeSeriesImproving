"""Frozen experiment configuration for the residual-headroom system.

The camelCase JSON mapping and the validation rules are binding; see
docs/headroom-contracts.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


# Feedback arms; the run's arm is a loop argument, not a config field.
# "augmented" = the fixed card PLUS the agent's own validated probes.
ARM_VALUES = ("scalar", "fixed", "self-built", "augmented", "placebo")

# Spacing of the beyond-window lookback depths: L, L+16, L+32, ... (contract §5).
BEYOND_L_STRIDE = 16

_FIELD_ALIASES: dict[str, str] = {
    "input_length": "inputLength",
    "horizon": "horizon",
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
    "evolution": "evolution",
    "seed": "seed",
}
_JSON_ALIASES: dict[str, str] = {value: key for key, value in _FIELD_ALIASES.items()}


@dataclass(frozen=True)
class HeadroomConfig:
    """All knobs of one experiment run; see headroom-contracts.md §2."""

    input_length: int = 96
    horizon: int = 24
    # Diversity judge (evolution guidance): identical in every arm so the arm
    # contrast stays clean; off only in the evolution=false ablation.
    evolution: bool = True
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

    @property
    def max_lag(self) -> int:
        return self.lag_segments[-1][1]

    @property
    def burn_in(self) -> int:
        """Origins before this index cannot have fully realized features."""
        return self.max_lag + self.fir_len_cap

    def validate(self) -> list[str]:
        """Only the rules whose violation would silently corrupt results.

        Feature depths beyond the burn-in would make numpy negative indexing
        wrap around to future rows; overlapping lag segments would double-count
        features; split ratios that leave no test segment break the layout.
        """
        errors: list[str] = []
        depths = {
            "inputLength": self.input_length,
            "beyond-l feature depth": self.input_length + BEYOND_L_STRIDE * (self.beyond_l_lags - 1),
            "horizon feature depth": self.horizon + self.horizon_realized_lags - 1,
        }
        for name, depth in depths.items():
            if self.burn_in < depth:
                errors.append(f"burnIn (maxLag + firLenCap = {self.burn_in}) must be >= the {name} ({depth})")
        band_lag = self.band_period_range[1] // 4
        if band_lag > self.max_lag:
            errors.append(f"band period range implies a band lag of {band_lag} > maxLag ({self.max_lag})")
        previous_end = -1
        for k1, k2 in self.lag_segments:
            if not previous_end < k1 <= k2:
                errors.append(f"lagSegments must be sorted, non-overlapping, k1 <= k2; got ({k1}, {k2})")
            previous_end = k2
        if self.train_ratio + self.diag_ratio + self.cert_ratio >= 1.0:
            errors.append("trainRatio + diagRatio + certRatio must be < 1 (the rest is test)")
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
