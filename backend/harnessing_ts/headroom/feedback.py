from __future__ import annotations

"""Markdown feedback rendering for the scalar / fixed / self-built / placebo arms.

All four arms share one byte-identical scaffold and differ only in the
middle block, so any measured behavioural difference between arms is
attributable to that block; see docs/headroom-contracts.md §9 and the
coding plan §2.5/§3.2 (arm-neutral wording: no dataset or domain
vocabulary, no status language such as "certified").

``fixed``/``placebo`` render the backend
defect card (placebo = the same card with coordinate rows permuted);
``self-built`` renders the outputs of the agent's OWN validated probes —
the backend never injects diagnostic content into that arm, it only runs
and reports the agent's tools.
"""

import json
from typing import Any

import numpy as np

from harnessing_ts.headroom.config import ARM_VALUES, HeadroomConfig
from harnessing_ts.headroom.correctors import GainReport

# Probe output rendering: the probe contract bounds one probe's serialized
# output at 4 KB (experiments/probes.py OUTPUT_CAP_BYTES); the cap applies
# to the escaped cell text, where escaping can at most double the characters,
# so 8192 renders any contract-legal output uncut.
_PROBE_OUTPUT_CHARS = 8192
# Only the newest probes render; each rendered row shows its full output.
_PROBE_MAX_ROWS = 8


def _fmt(value: Any) -> str:
    """Render one number at 4 decimals; non-finite values become 'n/a'."""
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    if number == 0.0:
        number = 0.0  # collapse -0.0 so the appendix never shows "-0.0000"
    return f"{number:.4f}"


def render_feedback(
    arm: str,
    incumbent: dict[str, Any],
    gains: list[GainReport],
    history: list[dict[str, Any]],  # unused: round.md owns the history table
    cfg: HeadroomConfig,
    placebo_perm: tuple[int, ...] | None = None,
    probe_rows: list[dict[str, Any]] | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Render one round of agent feedback as markdown (contract §9).

    The scaffold (title with round and remaining budget, incumbent diagnostic
    MSE, per-horizon-group MSE, candidate history) is byte-identical across
    arms; only the middle block differs.  ``scalar`` gets no middle block,
    ``fixed``/``placebo`` get the defect card (the card wording is "remaining
    improvement (lower bound)", never "headroom"), ``self-built`` gets its own
    validated probes' outputs plus the latest probe validation results.

    Card ordering: coordinate rows are sorted by the ``lcb`` values the
    caller passed.  This function never computes or adjusts intervals; the
    caller must compute the card gains at the Bonferroni-adjusted level
    ``correctors.card_level(cfg, period_peaks, n_channels)`` (the A-M4
    winner's-curse adjustment) and pass the resulting reports here.  NaN lcb
    values sort last; ties keep the caller's order.  The ``total`` gain
    becomes the aggregate row and ``mu2`` the level row; hgroup gains are
    certification statistics and are excluded.

    Placebo: exactly the card layout with the coordinate rows permuted —
    each row keeps its own numbers, only the order changes.  With
    ``placebo_perm=None`` the permutation is
    ``np.random.default_rng(cfg.seed).permutation(len(rows))``, i.e. fixed
    per (seed, row count) and independent of global RNG state.

    Probe rows (self-built arm only) are dicts from the loop: ``probeId``,
    ``measures``, ``output`` (JSON-serializable dict) or ``error``.  Rows
    arrive oldest-first and only the NEWEST ``_PROBE_MAX_ROWS`` render.  The
    static how-to-write-probes guide is NOT rendered here — the loop appends
    it (it is prompt text, and it lives in experiments/prompts/probes.md).
    Validation rows carry ``probeId``, ``ok``, ``detail``.
    """
    if arm not in ARM_VALUES:
        raise ValueError(f"arm must be one of {ARM_VALUES}, got {arm!r}")
    blocks = [_scaffold_head(incumbent), _scaffold_groups(incumbent)]
    if arm in ("fixed", "placebo"):
        rows, total = _split_gains(gains)
        ordered = _sorted_by_lcb(rows)
        if arm == "placebo":
            perm = (
                tuple(placebo_perm)
                if placebo_perm is not None
                else _default_perm(len(ordered), cfg)
            )
        else:
            perm = None
        blocks.append(_card_table(ordered, total, perm))
    elif arm == "self-built":
        blocks.append(_probe_table(probe_rows or [], validation_rows or []))
    elif arm == "augmented":
        # The fixed card first (what the backend already covers), then the
        # agent's own validated probes (the agent's additions on top).
        rows, total = _split_gains(gains)
        blocks.append(_card_table(_sorted_by_lcb(rows), total, None))
        blocks.append(_probe_table(probe_rows or [], validation_rows or []))
    return "\n\n".join(blocks) + "\n"


def _scaffold_head(incumbent: dict[str, Any]) -> str:
    """Title line (round + remaining budget) and the incumbent diagnostic MSE."""
    return (
        f"# Improvement round {int(incumbent['round'])} — "
        f"{int(incumbent['budgetLeft'])} candidates remaining\n\n"
        f"Incumbent diagnostic MSE: {_fmt(incumbent['diagMse'])}"
    )


def _scaffold_groups(incumbent: dict[str, Any]) -> str:
    """MSE per horizon group; dict order is the caller's display order."""
    lines = ["## MSE by horizon group", "", "| group | mse |", "| --- | --- |"]
    for group, mse in incumbent["mseByHGroup"].items():
        lines.append(f"| {group} | {_fmt(mse)} |")
    return "\n".join(lines)



def _cell(text: Any) -> str:
    """Markdown-table-safe cell text: pipes are escaped, newlines become spaces."""
    return " ".join(str(text).replace("|", "\\|").splitlines())


def _probe_table(
    probe_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> str:
    """Self-built arm middle block: the agent's own tools' latest outputs.

    Only the agent's validated probes appear here — the fixed card's content
    is never injected into this arm, so a measured win of self-built over
    fixed is a win of agent-built tooling, not of extra backend hints.
    """
    sections: list[str] = []
    if validation_rows:
        lines = ["## Probe validation results", "", "| probe | verdict | detail |", "| --- | --- | --- |"]
        for row in validation_rows:
            verdict = "kept" if row.get("ok") else "rejected"
            lines.append(
                f"| {_cell(row['probeId'])} | {verdict} | {_cell(row.get('detail', ''))} |"
            )
        sections.append("\n".join(lines))
    lines = ["## Your probes on the incumbent", ""]
    if not probe_rows:
        lines.append(
            "No validated probes yet: this round you see only the scalar numbers. "
            "Write a probe (see the probe guide below) and once it passes "
            "validation, its output on every future incumbent appears here."
        )
    else:
        lines += ["| probe | signal | output |", "| --- | --- | --- |"]
        # Rows arrive in library order (oldest first); show the NEWEST probes
        # — the agent's latest tools are the ones it is iterating on, and the
        # spec freezes "8 newest rows".
        for row in probe_rows[-_PROBE_MAX_ROWS:]:
            if "error" in row and row["error"]:
                output = f"ERROR: {row['error']}"
            else:
                output = _cell(json.dumps(_probe_safe(row.get("output")), ensure_ascii=False))
                if len(output) > _PROBE_OUTPUT_CHARS:
                    output = output[:_PROBE_OUTPUT_CHARS] + "… (truncated)"
            lines.append(f"| {_cell(row['probeId'])} | {_cell(row.get('measures', ''))} | {output} |")
        if len(probe_rows) > _PROBE_MAX_ROWS:
            lines.append(f"({len(probe_rows) - _PROBE_MAX_ROWS} earlier probes not shown)")
    sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _probe_safe(value: Any) -> Any:
    """JSON-renderable view of a probe output.

    Probe outputs reach the loop through a JSON round-trip (the isolated
    worker), so only plain Python arrives; the one live conversion is the
    worker's bare NaN token, which strict JSON readers reject -> null.
    """
    if isinstance(value, dict):
        return {key: _probe_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_probe_safe(item) for item in value]
    if isinstance(value, float):
        return None if np.isnan(value) else value
    return value


def _split_gains(gains: list[GainReport]) -> tuple[list[GainReport], GainReport | None]:
    """Card rows (all coordinates) plus the total report.

    hgroup gains are excluded: the contract reserves them for certification,
    not for the coordinate table the card renders.
    """
    rows: list[GainReport] = []
    total: GainReport | None = None
    for gain in gains:
        kind = gain.coordinate.kind
        if kind == "total":
            total = gain
        elif kind != "hgroup":
            rows.append(gain)
    return rows, total


def _sorted_by_lcb(rows: list[GainReport]) -> list[GainReport]:
    """Descending by the caller's lcb (computed at correctors.card_level upstream)."""
    return sorted(
        rows,
        key=lambda gain: float(gain.lcb) if np.isfinite(gain.lcb) else -np.inf,
        reverse=True,
    )


def _default_perm(n: int, cfg: HeadroomConfig) -> tuple[int, ...]:
    """Deterministic permutation from cfg.seed alone; no global RNG state."""
    return tuple(int(i) for i in np.random.default_rng(cfg.seed).permutation(n))


def _card_table(
    ordered: list[GainReport], total: GainReport | None, perm: tuple[int, ...] | None
) -> str:
    """Defect card: coordinate rows, then the aggregate total, then mu2."""
    if perm is not None:
        if sorted(perm) != list(range(len(ordered))):
            raise ValueError(
                f"placebo_perm must be a permutation of 0..{len(ordered) - 1}, got {tuple(perm)}"
            )
        ordered = [ordered[i] for i in perm]
    lines = [
        "## Remaining improvement by coordinate",
        "",
        "| aspect | coordinate | delta | lower bound | strongest at h |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [_card_row(gain) for gain in ordered]
    if total is not None:
        # The total is not one defect candidate but the aggregate bound, so
        # it is reported after the coordinates rather than ranked among them.
        lines.append(_card_row(total))
        # mu2 = (mean residual)^2 is the realized gain of a constant level
        # fix, so it fills the delta column; no interval is available for it.
        lines.append(f"| level-bias | mu2 | {_fmt(total.mu2)} | n/a | n/a |")
    return "\n".join(lines)


def _card_row(gain: GainReport) -> str:
    window = _peak_range(gain.by_horizon)
    strongest = "n/a" if window is None else f"h in [{window[0]}, {window[1]}]"
    return (
        f"| {gain.coordinate.aspect} | {gain.coordinate.label} | "
        f"{_fmt(gain.delta)} | {_fmt(gain.lcb)} | {strongest} |"
    )


def _peak_range(by_horizon: np.ndarray) -> tuple[int, int] | None:
    """1-based inclusive horizon band around the argmax of ``by_horizon``.

    The band widens while the per-horizon gain stays above half its maximum
    (an FWHM-style window).  NaN entries act as barriers: an unknown value
    cannot be claimed as part of the band.  None means "nothing to say".
    """
    values = np.asarray(by_horizon, dtype=np.float64)
    if values.size == 0:
        return None
    masked = np.where(np.isfinite(values), values, -np.inf)
    top = int(np.argmax(masked))
    if not np.isfinite(masked[top]):
        return None
    threshold = 0.5 * masked[top]
    lo = hi = top
    while lo > 0 and masked[lo - 1] >= threshold:
        lo -= 1
    while hi + 1 < values.size and masked[hi + 1] >= threshold:
        hi += 1
    return lo + 1, hi + 1
