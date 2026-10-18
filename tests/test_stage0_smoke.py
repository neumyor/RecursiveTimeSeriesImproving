from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_stage0():
    source = _REPO_ROOT / "experiments" / "stage0.py"
    spec = importlib.util.spec_from_file_location("stage0_experiment", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules[cls.__module__],
    # so the module must be registered before execution.
    sys.modules["stage0_experiment"] = module
    spec.loader.exec_module(module)
    return module


def test_stage0_smoke_report_structure_and_core_gates() -> None:
    """Tiny-size end-to-end run: structure plus the size-independent gates.

    G0-2 (CI coverage) and G0-3+4 (tightness/localization) need full sizes;
    at smoke sizes their stats are only checked for shape.
    """
    stage0 = _load_stage0()
    report = stage0.run(smoke=True)
    assert set(report["gates"]) == {"G0-7", "G0-1", "G0-2", "G0-3+4", "G0-5", "G0-6"}
    for name, gate in report["gates"].items():
        assert isinstance(gate["pass"], bool), name
    # The analytic control holds at any size.  G0-1's saturation verdict is
    # sample-size dependent (the certified UCB widens at the small cert
    # sizes), so smoke only checks structure; the full-size run is the gate.
    assert report["gates"]["G0-7"]["pass"] is True
    assert report["gates"]["G0-1"]["stats"]["cells"] == 6
    assert report["gates"]["G0-5"]["pass"] is True
    stats = report["gates"]["G0-2"]["stats"]
    assert all(pair["reps"] >= 1 for pair in stats.values())
    localization = report["gates"]["G0-3+4"]["stats"]
    assert localization["localization_cells"] == len(stage0.EXPECTED_ASPECTS)
    assert 0.0 <= localization["localization_rate"] <= 1.0
    assert "all_pass" in report



def test_stage0_deep_smoke_report_structure() -> None:
    """Tiny-size deep recalibration: the v2 report's wiring, not its verdicts.

    G0-3+4 and G0-6 only mean something at full sizes (the v1 smoke test
    skips the same gates for the same reason); what is checked here is the
    deep mode itself: the recalibrated gates alone, the pre-registered deep
    pairs, the carry-over record, and the P2 screening block.
    """
    stage0 = _load_stage0()
    report = stage0.run_deep(smoke=True)
    assert set(report["gates"]) == {"G0-3+4", "G0-6"}
    for gate in report["gates"].values():
        assert isinstance(gate["pass"], bool)
    localization = report["gates"]["G0-3+4"]["stats"]
    assert localization["localization_cells"] == len(stage0.DEEP_EXPECTED_ASPECTS)
    assert 0.0 <= localization["localization_rate"] <= 1.0
    screening = report["gates"]["G0-6"]["stats"]
    assert screening["incumbent_model"] == "m0-patchtst"
    assert set(screening["remaining_gap_ratio_by_task"]) == set(stage0.SCREENING_TASKS)
    assert report["carry_over"].keys() == {"G0-1", "G0-2", "G0-5", "G0-7"}
    for name, entry in report["carry_over"].items():
        # the model-agnostic verdicts are carried from the frozen
        # fixed-level artifact, not re-run here
        assert entry["artifact"] == "experiments/specs/g0-final-report-v3.json"
        assert entry["pass"] is True, name
    assert set(report["p2_v2_screening"]["by_task"]) == set(stage0.P2_V2_TASKS)
    assert all(entry["seeds"] == 1 for entry in report["p2_v2_screening"]["by_task"].values())
    assert "all_pass" in report
