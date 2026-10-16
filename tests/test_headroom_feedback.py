from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.feedback import render_feedback

HORIZON = 24
ARMS = ("scalar", "fixed", "self-built", "augmented", "placebo")
# Contract/plan §3.2 forbidden vocabulary: dataset names, domain words, and
# status language that could prime or certify any arm.
FORBIDDEN_SUBSTRINGS = (
    "ecg",
    "ett",
    "electricity",
    "traffic",
    "weather",
    "exchange",
    "headroom",
    "certified",
    "certificate",
    "saturated",
)


@dataclass(frozen=True)
class _Coordinate:
    """Stand-in for correctors.Coordinate; feedback duck-types at runtime."""

    kind: str
    label: str
    aspect: str
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Gain:
    """Stand-in for correctors.GainReport; the renderer duck-types at runtime."""

    coordinate: _Coordinate
    delta: float
    lcb: float
    ucb: float
    by_horizon: np.ndarray
    mu2: float
    n_eval: int


def _make_gains() -> list[_Gain]:
    def by_horizon(delta: float, peak: int) -> np.ndarray:
        # Tent profile over horizons; only the peak position matters here.
        h = np.arange(1, HORIZON + 1, dtype=np.float64)
        return np.maximum(0.0, 1.0 - np.abs(h - peak) / 6.0) * delta

    specs = [
        ("band", "band(P=24)", "periodicity", 0.034, 0.026, 2),
        ("lag", "lag[0,23]", "lag-structure", 0.021, 0.015, 1),
        ("beyond_l", "beyond_l", "long-range-beyond-window", 0.015, 0.009, 20),
        ("channel", "channel->3", "cross-channel", 0.006, 0.002, 5),
        ("horizon", "horizon", "horizon-range", 0.004, 0.001, 12),
    ]
    gains = [
        _Gain(_Coordinate(kind, label, aspect), delta, lcb, lcb + 0.004, by_horizon(delta, peak), 0.0004, 500)
        for kind, label, aspect, delta, lcb, peak in specs
    ]
    gains.append(
        _Gain(_Coordinate("total", "total", "other"), 0.061, 0.047, 0.075, by_horizon(0.061, 4), 0.0004, 500)
    )
    # hgroup gains exist for certification; the card must not render them.
    gains.append(
        _Gain(
            _Coordinate("hgroup", "hgroup[1,12]", "horizon-range"),
            0.010,
            0.005,
            0.014,
            by_horizon(0.010, 6),
            0.0004,
            500,
        )
    )
    return gains


def _make_incumbent() -> dict[str, Any]:
    return {
        "diagMse": 0.4123,
        "mseByHGroup": {"hgroup[1,12]": 0.4012, "hgroup[13,24]": 0.4234},
        "round": 3,
        "budgetLeft": 17,
    }


def _make_history(entries: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "round": i + 1,
            "candidateId": f"c{i + 1:02d}",
            "accepted": i % 2 == 0,
            "diagMse": 0.4300 + 0.001 * i,
        }
        for i in range(entries)
    ]


def _card_rows(text: str) -> list[str]:
    """Data rows of the defect-card table, in rendered order."""
    section = text.split("## Remaining improvement by coordinate", 1)[1]
    return [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| aspect")
    ]


def test_scalar_is_exactly_the_scaffold() -> None:
    out = render_feedback(
        "scalar", _make_incumbent(), _make_gains(), _make_history(2), HeadroomConfig()
    )
    assert out.startswith("# Improvement round 3 — 17 candidates remaining\n\n")
    assert "Incumbent diagnostic MSE: 0.4123" in out
    assert "| group | mse |" in out
    assert "| hgroup[1,12] | 0.4012 |" in out
    assert "| hgroup[13,24] | 0.4234 |" in out
    # No middle block of any other arm leaks in.
    assert "aspect" not in out
    assert "Remaining improvement" not in out
    assert "Your probes" not in out


def test_all_arms_share_the_identical_scaffold() -> None:
    incumbent = _make_incumbent()
    history = _make_history(3)
    gains = _make_gains()
    scalar = render_feedback("scalar", incumbent, gains, history, HeadroomConfig())
    # The scaffold pieces, extracted from the scalar render: the header and
    # the horizon-group table must appear verbatim in every arm.
    head, sep, rest = scalar.partition("## MSE by horizon group")
    assert sep
    chunks = (head, "## MSE by horizon group" + rest)
    for arm in ARMS:
        out = render_feedback(arm, incumbent, gains, history, HeadroomConfig())
        for chunk in chunks:
            assert chunk in out, arm
        # scalar is the only arm whose output is the bare scaffold.
        assert (out == scalar) == (arm == "scalar")


def test_fixed_orders_coordinates_by_lcb_descending() -> None:
    out = render_feedback("fixed", _make_incumbent(), _make_gains(), _make_history(1), HeadroomConfig())
    section = out.split("## Remaining improvement by coordinate", 1)[1]
    expected = ["band(P=24)", "lag[0,23]", "beyond_l", "channel->3", "horizon"]
    positions = [section.index(f"| {label} |") for label in expected]
    assert positions == sorted(positions)
    # The aggregate total and mu2 rows come after every coordinate row.
    assert max(positions) < section.index("| other | total | 0.0610 |")
    assert section.index("| other | total | 0.0610 |") < section.index(
        "| level-bias | mu2 | 0.0004 | n/a | n/a |"
    )
    # hgroup gains feed certification only: the scaffold MSE table is the
    # sole place the group label appears.
    assert out.count("hgroup[1,12]") == 1


def test_fixed_never_renders_card_content_into_self_built() -> None:
    """The self-built arm's diagnostic channel is the agent's own probes only;
    any fixed-card content leaking in would contaminate the core contrast."""
    out = render_feedback(
        "self-built",
        _make_incumbent(),
        _make_gains(),
        _make_history(1),
        HeadroomConfig(),
        probe_rows=[{"probeId": "p1", "measures": "dominant period", "output": {"period": 24.0}}],
    )
    assert "Remaining improvement by coordinate" not in out
    assert "band(P=24)" not in out
    assert "lower bound" not in out


def test_placebo_renders_the_same_rows_in_a_permuted_order() -> None:
    cfg = HeadroomConfig()
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    fixed_rows = _card_rows(render_feedback("fixed", incumbent, gains, history, cfg))
    placebo_rows = _card_rows(render_feedback("placebo", incumbent, gains, history, cfg))
    assert sorted(placebo_rows) == sorted(fixed_rows)  # same multiset of rows
    assert placebo_rows != fixed_rows  # seed-0 default permutation moves them
    assert placebo_rows[-2:] == fixed_rows[-2:]  # total and mu2 stay pinned at the end


def test_explicit_placebo_perm_is_honored() -> None:
    cfg = HeadroomConfig()
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    fixed_rows = _card_rows(render_feedback("fixed", incumbent, gains, history, cfg))
    perm = (4, 2, 0, 3, 1)
    placebo_rows = _card_rows(
        render_feedback("placebo", incumbent, gains, history, cfg, placebo_perm=perm)
    )
    assert placebo_rows[:-2] == [fixed_rows[i] for i in perm]
    assert placebo_rows[-2:] == fixed_rows[-2:]


def test_invalid_placebo_perm_is_rejected() -> None:
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    for bad in ((0, 0, 1, 2, 3), (0, 1, 2), (0, 1, 2, 3, 5)):
        with pytest.raises(ValueError, match="permutation"):
            render_feedback(
                "placebo", incumbent, gains, history, HeadroomConfig(), placebo_perm=bad
            )


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="arm"):
        render_feedback("bogus", _make_incumbent(), [], _make_history(1), HeadroomConfig())


def test_no_forbidden_words_in_any_arm() -> None:
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(2)
    for arm in ARMS:
        out = render_feedback(
            arm,
            incumbent,
            gains,
            history,
            HeadroomConfig(),
            probe_rows=[{"probeId": "p1", "measures": "m", "output": {"x": 1.0}}],
            validation_rows=[{"probeId": "p1", "ok": True, "detail": "validated"}],
        )
        for word in FORBIDDEN_SUBSTRINGS:
            assert word not in out.lower(), (arm, word)


def test_history_table_is_owned_by_round_md() -> None:
    """round.md renders the candidate history; the feedback scaffold no longer
    duplicates it (one table in the prompt, byte-identical across arms)."""
    out = render_feedback("fixed", _make_incumbent(), _make_gains(), _make_history(15), HeadroomConfig())
    assert "| 15 | c15 |" not in out
    assert "## Candidate history" not in out


def test_self_built_without_probes_shows_the_bootstrap_message() -> None:
    out = render_feedback(
        "self-built", _make_incumbent(), [], _make_history(1), HeadroomConfig()
    )
    assert "## Your probes on the incumbent" in out
    assert "No validated probes yet" in out


def test_self_built_renders_probe_outputs_and_validation() -> None:
    out = render_feedback(
        "self-built",
        _make_incumbent(),
        [],
        _make_history(1),
        HeadroomConfig(),
        probe_rows=[
            {"probeId": "p-period", "measures": "dominant residual period", "output": {"period": 24.0}},
            {
                "probeId": "p-attn",
                "measures": "block0 activation norm",
                "error": "ProbeTimeout: ran past 120.0s",
            },
        ],
        validation_rows=[
            {"probeId": "p-period", "ok": True, "detail": "validated"},
            {"probeId": "p-bad", "ok": False, "detail": "signalKey missing from output"},
        ],
    )
    assert "## Probe validation results" in out
    assert "| p-period | kept | validated |" in out
    assert "| p-bad | rejected | signalKey missing from output |" in out
    assert '| p-period | dominant residual period | {"period": 24.0} |' in out
    assert "| p-attn | block0 activation norm | ERROR: ProbeTimeout: ran past 120.0s |" in out


def test_self_built_truncates_chatty_probe_output() -> None:
    chatty = {"series": list(range(400))}
    out = render_feedback(
        "self-built",
        _make_incumbent(),
        [],
        _make_history(1),
        HeadroomConfig(),
        probe_rows=[{"probeId": "p", "measures": "m", "output": chatty}],
    )
    assert "… (truncated)" in out
    assert len(out.split("| p | m | ", 1)[1].split(" |", 1)[0]) <= 620


def test_self_built_shows_the_newest_probe_rows() -> None:
    """The 8 NEWEST probes render (the agent iterates on its latest tools);
    the note counts the earlier ones that are hidden."""
    rows = [
        {"probeId": f"p{i:02d}", "measures": "m", "output": {"i": i}} for i in range(10)
    ]
    out = render_feedback(
        "self-built", _make_incumbent(), [], _make_history(1), HeadroomConfig(), probe_rows=rows
    )
    assert "| p02 |" in out and "| p09 |" in out
    assert "| p00 |" not in out and "| p01 |" not in out
    assert "2 earlier probes not shown" in out


def test_augmented_renders_card_and_probe_outputs_together() -> None:
    """The augmented arm's middle block is the fixed card plus the agent's own
    probe outputs and verdicts — the full combination treatment."""
    out = render_feedback(
        "augmented",
        _make_incumbent(),
        _make_gains(),
        _make_history(1),
        HeadroomConfig(),
        probe_rows=[
            {"probeId": "p-nonlin", "measures": "threshold coupling", "output": {"r2": 0.2}}
        ],
        validation_rows=[{"probeId": "p-nonlin", "ok": True, "detail": "validated"}],
    )
    # The backend card is present, in the fixed arm's exact wording.
    assert "## Remaining improvement by coordinate" in out
    assert "| band(P=24) | 0.0340 | 0.0260 |" in out
    # The agent's probe channel rides below it.
    assert "## Probe validation results" in out
    assert "| p-nonlin | kept | validated |" in out
    assert '"r2": 0.2' in out
