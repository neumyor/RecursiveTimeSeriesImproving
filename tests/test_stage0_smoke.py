from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_stage0():
    source = _REPO_ROOT / "experiments" / "stage0.py"
    spec = importlib.util.spec_from_file_location("stage0_experiment", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
    # Analytic control and the two oracle-backed gates must hold at any size.
    assert report["gates"]["G0-7"]["pass"] is True
    assert report["gates"]["G0-1"]["pass"] is True
    assert report["gates"]["G0-5"]["pass"] is True
    stats = report["gates"]["G0-2"]["stats"]
    assert all(pair["reps"] >= 1 for pair in stats.values())
    localization = report["gates"]["G0-3+4"]["stats"]
    assert localization["localization_cells"] == 7
    assert 0.0 <= localization["localization_rate"] <= 1.0
    assert "all_pass" in report
