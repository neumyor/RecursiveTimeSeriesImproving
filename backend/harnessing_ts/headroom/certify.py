"""Epsilon-saturation check for the incumbent corrector; see docs/headroom-contracts.md §7.

The rule is one sentence: an incumbent is saturated when, on the cert
segment, EVERY statistic's realized-gain upper confidence bound sits below
``certify_epsilon`` times that statistic's group MSE — no residual coordinate
retains a relative improvement beyond epsilon.  ``gains`` must be pre-computed
at the fixed level ``1 - cfg.certify_alpha`` via
``correctors.all_gains(..., level=...)``.  The mean CI is built
two-sided, so each UCB actually sits at the one-sided level
``1 - certify_alpha / 2`` — a deliberate factor-of-two conservatism in the
stopping direction.  There is no cross-version or cross-statistic alpha
accounting; the false-stop rate is an empirical property, measured by G0-5.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import GainReport

@dataclass(frozen=True)
class CertReport:
    """Result of one saturation check; field shapes are binding (§7)."""

    saturated: bool
    level: float  # per-statistic confidence level (fixed)
    failures: list[dict[str, str | float]]  # {"label", "ucb", "threshold"} per violating gain


def certify(gains: list[GainReport], mse_by_group: dict[str, float], cfg: HeadroomConfig) -> CertReport:
    """Decide whether the incumbent is epsilon-saturated on the cert segment.

    The caller invokes certify only when the incumbent changed (round 0 and
    after each acceptance).  Saturation means every statistic's gain UCB lies
    strictly below certify_epsilon times its group's cert-segment MSE.
    """
    if not gains:
        raise ValueError("certify requires at least one gain, got an empty list")
    missing = sorted({gain.coordinate.label for gain in gains} - set(mse_by_group))
    if missing:
        raise ValueError(f"certify requires mse_by_group to cover every gain label, missing {missing}")

    level = 1.0 - cfg.certify_alpha

    failures: list[dict[str, str | float]] = []
    for gain in gains:
        label = gain.coordinate.label
        threshold = cfg.certify_epsilon * mse_by_group[label]
        # The stop rule is a strict "<": an UCB exactly at epsilon still leaves
        # room for an epsilon-improvement, so equality counts as a violation.
        if gain.ucb >= threshold:
            failures.append({"label": label, "ucb": gain.ucb, "threshold": threshold})
    return CertReport(saturated=not failures, level=level, failures=failures)
