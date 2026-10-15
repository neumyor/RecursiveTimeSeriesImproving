"""Epsilon-saturation certificate for the incumbent corrector; see docs/headroom-contracts.md §7.

``gains`` must be pre-computed at the Bonferroni level returned by
``correctors.certify_level(cfg, j_max, period_peaks, n_channels)`` and passed
through ``correctors.all_gains(..., level=...)``; certify() rejects gains
carrying any other level, because a UCB read off-level silently voids the
family-wise bound.  That level is two-sided — ``mean_ci`` builds one two-sided
interval per statistic — so each UCB compared below actually sits at the
one-sided level 1 - alpha_j / (2K) and the family-wise false-early-stop
probability is bounded by certify_alpha / 2, not certify_alpha.  The
factor-of-two conservatism is deliberate: it can only make stopping harder,
never too eager.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import GainReport

# Levels agree only when the caller wired the same Bonferroni arithmetic;
# anything inside this tolerance is float noise, not a wiring bug.
_LEVEL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CertReport:
    """Result of one certification check; field shapes are binding (§7)."""

    round_index: int  # j-th certification, counted from 1
    saturated: bool
    level: float  # per-statistic confidence level 1 - alpha_j / K
    failures: list[dict[str, str | float]]  # {"label", "ucb", "threshold"} per violating gain


def certify(
    gains: list[GainReport],
    mse_by_group: dict[str, float],
    cfg: HeadroomConfig,
    j: int,
    j_max: int,
) -> CertReport:
    """Decide whether the incumbent is epsilon-saturated on the cert segment.

    The caller invokes certify only when the incumbent changed (round 0 and
    after each acceptance), so j counts incumbent versions rather than raw
    candidate rounds, and the total certify_alpha budget is spent uniformly
    over at most j_max certifications: alpha_j = certify_alpha / j_max. Within
    one check, Bonferroni splits alpha_j across the K statistics, so every
    gain must carry the level 1 - alpha_j / K (see the module docstring for
    the two-sided reading and its alpha/2 conservatism). Saturation means
    every coordinate's gain UCB lies below certify_epsilon times its group's
    cert-segment MSE: no aspect retains a relative improvement beyond
    epsilon.
    """
    if j < 1:
        raise ValueError(f"certify requires a 1-based round index j >= 1, got {j}")
    if j_max < j:
        raise ValueError(f"certify requires j_max >= j, got j_max={j_max} < j={j}")
    if not gains:
        raise ValueError("certify requires at least one gain, got an empty list")
    missing = sorted({gain.coordinate.label for gain in gains} - set(mse_by_group))
    if missing:
        raise ValueError(f"certify requires mse_by_group to cover every gain label, missing {missing}")

    # A larger planned j_max leaves each check a smaller alpha share, hence a
    # higher per-statistic level and wider UCBs: planning more certifications
    # makes each one individually stricter so the total stays under certify_alpha.
    level = 1.0 - (cfg.certify_alpha / j_max) / len(gains)

    # Regression guard for the level wiring: an off-level UCB breaks the
    # family-wise bound without any visible symptom, so refuse to certify.
    # Written as `not <=` so a NaN level is rejected too.
    for gain in gains:
        if not abs(float(gain.level) - level) <= _LEVEL_TOLERANCE:
            raise RuntimeError(
                f"gain {gain.coordinate.label!r} was computed at level {gain.level}, but "
                f"certification j={j} of j_max={j_max} with K={len(gains)} statistics "
                f"requires level {level}; pre-compute the gains at the Bonferroni "
                f"level via correctors.all_gains(..., level=correctors.certify_level("
                f"cfg, j_max, period_peaks, n_channels))"
            )

    failures: list[dict[str, str | float]] = []
    for gain in gains:
        label = gain.coordinate.label
        threshold = cfg.certify_epsilon * mse_by_group[label]
        # The stop rule is a strict "<": an UCB exactly at epsilon still leaves
        # room for an epsilon-improvement, so equality counts as a violation.
        if gain.ucb >= threshold:
            failures.append({"label": label, "ucb": gain.ucb, "threshold": threshold})
    return CertReport(round_index=j, saturated=not failures, level=level, failures=failures)
