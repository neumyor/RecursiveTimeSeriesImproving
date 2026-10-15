"""Epsilon-saturation certificate for the incumbent corrector; see docs/headroom-contracts.md §7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harnessing_ts.headroom.config import HeadroomConfig

if TYPE_CHECKING:
    # correctors.py is coded in parallel; at runtime certify only touches the
    # duck-typed surface `.coordinate.label` / `.ucb`, so this import never runs.
    from harnessing_ts.headroom.correctors import GainReport


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
    one check, Bonferroni splits alpha_j across the K statistics, so each UCB
    is read at level 1 - alpha_j / K; the union bound then keeps the
    family-wise false-early-stop probability below certify_alpha. Saturation
    means every coordinate's gain UCB lies below certify_epsilon times its
    group's cert-segment MSE: no aspect retains a relative improvement beyond
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

    failures: list[dict[str, str | float]] = []
    for gain in gains:
        label = gain.coordinate.label
        threshold = cfg.certify_epsilon * mse_by_group[label]
        # The stop rule is a strict "<": an UCB exactly at epsilon still leaves
        # room for an epsilon-improvement, so equality counts as a violation.
        if gain.ucb >= threshold:
            failures.append({"label": label, "ucb": gain.ucb, "threshold": threshold})
    return CertReport(round_index=j, saturated=not failures, level=level, failures=failures)
