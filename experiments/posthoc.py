from __future__ import annotations

"""Offline analysis: finished loop runs -> the paper's numbers (final plan §4).

One run = one (task, arm, seed) cell produced by experiments/loop.py.  This
script never talks to an LLM: it replays each run's incumbent chain offline
(re-fit every accepted candidate on train, predict the held-out test
segment), applies the uniform m+g postprocessing to every incumbent, and
scores the pre-registered endpoints (experiments/specs/main.json):

- E1  normalized test optimality gap of the FINAL incumbent after the m+g
      postprocessing, ngap = gap(final)/gap(m0); real tasks (no oracle)
      report the relative MSE ratio MSE(final)/MSE(m0) instead.
- E2  AUC of the same normalized value over the run's rounds: the mean of
      "the incumbent's value after round r" for r = 1..budget.

Per run it also reports the diagnostics (acceptance/failure counts, tokens,
wall-clock, numpy-vs-torch mix, directed/off-target rates against the
parent's top card coordinate, the cert-level total LCB vs true-test-gap
correlation across accepted incumbents, the certification stop point and
the post-stop gain) and the audit: absolute paths outside the workspace
(and outside experiments/templates and backend) in the agent's read/glob/
grep tool calls — the heldout path appearing there voids the run's
isolation — plus candidates whose diag card gain dwarfs their cert gain
(ratio > 5), the overfit-to-diagnostic signature.

Arm level: per-(task, arm) median+IQR of E1/E2 and the seed-blocked
sign-flip permutation test for the three pre-registered contrasts
(self-built vs fixed, fixed vs scalar, self-built vs scalar), exact up to
8192 patterns and Monte Carlo (10k, fixed seed) beyond.

Usage:
    uv run --extra research python experiments/posthoc.py \
        --runs experiments/runs/p2 --out experiments/runs/p2/analysis.json \
        [--config experiments/specs/main-config.json] \
        [--real-csv real-ett=experiments/data/ETTh1.csv]

The replay config is the --config file (or HeadroomConfig defaults) with
inputLength/horizon taken from the run's own ws/meta.json and the seed
from the held-out reference, so a mismatched config cannot silently
mis-predict.  Real runs need --real-csv to map the run's task name to its
CSV: the loop never stores the path anywhere the agent could see it.
"""

import argparse
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterator

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import baselines
import common
import evaluate as ev
import realdata
from harnessing_ts.headroom.candidate import SourceRejected, check_source
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    all_gains,
    card_level,
    corrector_gain,
    fit_greedy,
    select_period_peaks,
)
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.timeidx import segment_origins, windows

# m0's source is the repo template: the loop copies it into
# ws/models/incumbent/model.py at prepare(), but every acceptance overwrites
# that file, so only the template still holds m0 verbatim.  The v2 default is
# the PatchTST-lite template; --m0-template overrides it for analyzing
# runs made with the linear template.
_DEFAULT_M0_TEMPLATE = "m0_deep.py"
# The only absolute roots a well-behaved agent may touch besides its own ws.
_ALLOWED_ROOTS = (_REPO_ROOT / "experiments" / "templates", _REPO_ROOT / "backend")
_READ_TOOLS = ("read", "glob", "grep")
_CONTRASTS = (("self-built", "fixed"), ("fixed", "scalar"), ("self-built", "scalar"))
_EXACT_PERM_CAP = 8192  # 2^13 sign patterns; at or below this the test is exact
_MC_PERMS = 10_000
_GAIN_RATIO_FLAG = 5.0


class _RunError(Exception):
    """A run whose artifacts cannot be analyzed; recorded, never raised past it."""


# --- JSON / module helpers (the loop's own disciplines) --------------------------------


_json_safe = common.json_safe
_read_json = common.read_json
_read_jsonl = common.read_jsonl
_load_module = common.load_module


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Best-effort JSONL walk for audit evidence (logs, not ledgers)."""
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def _gain_row(gain: Any) -> dict[str, Any]:
    """Serialize one GainReport exactly like the loop's heldout posthoc card."""
    return {
        "label": gain.coordinate.label,
        "aspect": gain.coordinate.aspect,
        "delta": gain.delta,
        "lcb": gain.lcb,
        "ucb": gain.ucb,
        "mu2": gain.mu2,
        "by_horizon": gain.by_horizon,
    }


def _top_aspect(rows: list[dict[str, Any]] | None) -> str | None:
    """Aspect of the card's top coordinate row (highest lcb, card ordering).

    The total row (the aggregate bound) and the two hgroup rows are
    certification statistics, not targetable defects, so they never rank;
    `max` returns the first maximum, matching feedback._sorted_by_lcb's
    stable sort on ties.
    """
    if not rows:
        return None

    def key(row: dict[str, Any]) -> float:
        lcb = row.get("lcb")
        return float(lcb) if isinstance(lcb, (int, float)) and math.isfinite(lcb) else -math.inf

    coordinates = [
        row
        for row in rows
        if row.get("label") != "total" and not str(row.get("label", "")).startswith("hgroup[")
    ]
    if not coordinates:
        return None
    return str(max(coordinates, key=key)["aspect"])


# --- per-run replay ---------------------------------------------------------------------


class _Replay:
    """Offline re-fit cache for one run: task arrays, config, candidate segments.

    Every candidate is re-fitted ONCE on train and scored on diag and cert.
    The loop's determinism gate makes accepted candidates reproducible;
    rejected ones only feed the audit, where nondeterminism would only add
    noise to a heuristic flag.  Test predictions are built only for
    incumbent-chain members — rejected candidates never reach test.
    """

    def __init__(
        self,
        run_dir: Path,
        base_cfg: HeadroomConfig,
        real_csvs: dict[str, Path],
        m0_template: Path,
    ) -> None:
        self.run_dir = run_dir
        self.ws = run_dir / "ws"
        heldout_ref = run_dir / "heldout.json"
        if not heldout_ref.is_file():
            raise _RunError("heldout.json is missing; not a loop run directory")
        self.heldout = Path(_read_json(heldout_ref)["path"])
        self.reference: dict[str, Any] = _read_json(self.heldout / "reference.json")
        meta: dict[str, Any] = _read_json(self.ws / "meta.json")
        self.task_name = str(self.reference["task"])
        self.seed = int(self.reference["seed"])
        # L/H come from the run's own meta (the candidate contract every
        # model was fitted under), not from the analysis config.
        self.cfg = replace(
            base_cfg, seed=self.seed, input_length=int(meta["L"]), horizon=int(meta["H"])
        )
        self.meta_fit = {
            "L": int(meta["L"]),
            "H": int(meta["H"]),
            "C": int(meta["C"]),
            "seed": self.seed,
        }
        self.is_real = meta.get("task") == "real-deidentified"
        self._load_task(real_csvs)
        self.m0_template = m0_template
        self._states: dict[str, dict[str, Any] | None] = {}
        self._cards: dict[str, list[dict[str, Any]] | None] = {}
        self.errors: dict[str, str] = {}

    def _load_task(self, real_csvs: dict[str, Path]) -> None:
        """Regenerate the run's task: the test/cert arrays and oracle never
        left the heldout side, so they are re-derived from the reference."""
        if self.is_real:
            csv = real_csvs.get(self.task_name)
            if csv is None:
                raise _RunError(f"real run needs --real-csv {self.task_name}=<csv path>")
            task = realdata.load_real_task(Path(csv), self.cfg)
            self.oracle = None
        else:
            if self.task_name not in REGISTRY:
                raise _RunError(f"unknown synthetic task {self.task_name!r}")
            sizes = {key: int(value) for key, value in self.reference["sizes"].items()}
            task = REGISTRY[self.task_name](seed=self.seed, sizes=sizes)
            self.oracle = task.oracle
        self.train_diag = task.train_diag
        self.diag_start = task.diag_start
        self.cert_values = task.cert
        self.test_values = task.test
        self.train = self.train_diag[: self.diag_start]
        # Integrity: the stored series must be the task the reference
        # regenerates, or every number below would describe a different
        # dataset than the one the agent saw (the in-run sha check only
        # covered the run's lifetime).
        series = self.ws / "data" / "series.npy"
        self.series_matches_task = (
            bool(np.array_equal(np.load(series), self.train_diag)) if series.is_file() else None
        )

    def candidate_state(self, cand_id: str) -> dict[str, Any] | None:
        """(cached) diag/cert segments + diag-A peaks of one re-fitted model."""
        if cand_id in self._states:
            return self._states[cand_id]
        state: dict[str, Any] | None
        try:
            source = (
                self.m0_template
                if cand_id == "m0"
                else self.ws / "models" / "candidates" / cand_id / "model.py"
            )
            if cand_id == "m0":
                # A wrong template silently recomputes every gap baseline
                # (ngap divides by gap(m0)).  The run's heldout side recorded
                # the template NAME the loop copied, so the replay loads that
                # exact file; --m0-template stays as an explicit override.
                recorded = _read_json(self.heldout / "incumbent-reference.json").get("template")
                if recorded:
                    source = _REPO_ROOT / "experiments" / "templates" / recorded
            if not source.is_file():
                raise _RunError(f"candidate source is missing: {source}")
            text = source.read_text(encoding="utf-8")
            try:
                check_source(text)  # the same AST gate the loop applied in-run
            except SourceRejected as exc:
                raise _RunError(f"source rejected by the candidate check: {exc}") from exc
            module = _load_module("rhg_posthoc_" + re.sub(r"\W+", "_", cand_id), source)
            adapter = module.create_adapter()
            adapter.fit(self.train, self.meta_fit)
            start = self.cfg.input_length - 1  # standalone segments begin one lookback in
            pred_diag = ev.predict_adapter(
                adapter, self.train_diag, self.diag_start, self.train_diag.shape[0], self.cfg
            )
            seg_diag = ev.build_segment(
                self.train_diag, self.diag_start, self.train_diag.shape[0], pred_diag, self.cfg
            )
            pred_cert = ev.predict_adapter(
                adapter, self.cert_values, start, self.cert_values.shape[0], self.cfg
            )
            seg_cert = ev.build_segment(
                self.cert_values, start, self.cert_values.shape[0], pred_cert, self.cfg
            )
            seg_a, _ = ev.split_ab(seg_diag, self.cfg)
            state = {
                "source": text,
                "seg_diag": seg_diag,
                "seg_cert": seg_cert,
                "peaks": select_period_peaks(seg_a.e_full, seg_a.origins, self.cfg),
                "adapter": adapter,
            }
        except Exception as exc:  # recorded per candidate; the run continues
            self.errors[cand_id] = str(exc)
            state = None
        self._states[cand_id] = state
        return state

    def test_state(self, cand_id: str) -> dict[str, Any] | None:
        """candidate_state plus the model's test segment and usable predictions."""
        state = self.candidate_state(cand_id)
        if state is None or "seg_test" in state:
            return state
        start = self.cfg.input_length - 1
        pred_test = ev.predict_adapter(
            state["adapter"], self.test_values, start, self.test_values.shape[0], self.cfg
        )
        seg_test = ev.build_segment(
            self.test_values, start, self.test_values.shape[0], pred_test, self.cfg
        )
        state["seg_test"] = seg_test
        state["pred_test"] = pred_test[seg_test.origins - start]
        state.pop("adapter")  # nothing re-predicts afterwards; free the memory
        return state

    def fitted_total(self, cand_id: str) -> Any | None:
        """(cached) the greedy total corrector fitted on the candidate's FULL diag.

        This is the fit the certification used (greedy on all of diag), so
        the cert-side gains and LCBs below are the certified quantities.
        """
        state = self.candidate_state(cand_id)
        if state is None:
            return None
        if "fitted" not in state:
            state["fitted"] = fit_greedy(state["seg_diag"], self.cfg, state["peaks"])
        return state["fitted"]

    def card(self, cand_id: str) -> list[dict[str, Any]] | None:
        """(cached) The model's card rows: heldout copy if present, else recomputed.

        The stored card is the parent's own residual state at the time the
        agent saw it (written when the parent was still a candidate) — that
        is exactly the state the next round's proposal reacts to, so it is
        preferred over a recomputation.  m0 was never a candidate and has no
        stored card, so its card is rebuilt from the replayed segments —
        cached, because every early round has m0 as its parent.
        """
        if cand_id in self._cards:
            return self._cards[cand_id]
        rows: list[dict[str, Any]] | None = None
        if cand_id != "m0":
            path = self.heldout / "posthoc" / f"{cand_id}.json"
            if path.is_file():
                loaded = _read_json(path).get("card")
                if isinstance(loaded, list):
                    rows = loaded
        if rows is None:
            state = self.candidate_state(cand_id)
            if state is not None:
                seg_a, seg_b = ev.split_ab(state["seg_diag"], self.cfg)
                channels = int(state["seg_diag"].values.shape[1])
                level = card_level(self.cfg, state["peaks"], channels)
                rows = [
                    _gain_row(gain)
                    for gain in all_gains(seg_a, seg_b, self.cfg, state["peaks"], level=level)
                ]
        self._cards[cand_id] = rows
        return rows


# --- audit ------------------------------------------------------------------------------


def _offending_paths(text: str, allowed: list[Path]) -> set[str]:
    """Absolute filesystem paths in ``text`` outside every allowed root."""
    found: set[str] = set()
    text = re.sub(r"https?://\S+", " ", text)  # URL path segments are not filesystem paths
    for match in re.findall(r"/[A-Za-z0-9_][A-Za-z0-9_.\-/]*", text):
        raw = match.rstrip(".,;:)]}\"'")
        resolved = Path(raw).resolve()
        if all(resolved != root and root not in resolved.parents for root in allowed):
            found.add(raw)
    return found


def _audit_paths(run_dir: Path) -> list[str]:
    """Out-of-workspace paths in the agent's read/glob/grep tool calls.

    The round logs are the MessageLog's Part records (``type: tool_call``,
    ``name``, ``input``); the DSH session logs are zstd JSONL whose
    ``type: "tool/call"`` records carry the arguments as a JSON string.  A
    missing zstandard only skips the second source, never fails the audit.
    """
    ws = run_dir / "ws"
    allowed = [root.resolve() for root in (ws, *_ALLOWED_ROOTS)]
    found: set[str] = set()
    for log in sorted((ws / "logs").glob("round-*.jsonl")):
        for part in _iter_json_objects(log.read_text(encoding="utf-8", errors="replace")):
            if part.get("name") in _READ_TOOLS and part.get("type") in ("tool_call", "tool_use"):
                found |= _offending_paths(json.dumps(part.get("input", {}), default=str), allowed)
    sessions = ws / ".dsh-sessions"
    if sessions.is_dir():
        try:
            import zstandard
        except ImportError:
            zstandard = None
        if zstandard is not None:
            dctx = zstandard.ZstdDecompressor()
            for zlog in sorted(sessions.rglob("session.jsonl.zstd")):
                with zlog.open("rb") as handle:
                    text = dctx.stream_reader(handle).read().decode("utf-8", "replace")
                for record in _iter_json_objects(text):
                    data = record.get("data")
                    if (
                        record.get("type") == "tool/call"
                        and isinstance(data, dict)
                        and data.get("name") in _READ_TOOLS
                    ):
                        found |= _offending_paths(
                            json.dumps(data.get("arguments", ""), default=str), allowed
                        )
    return sorted(found)


# --- per-run analysis -------------------------------------------------------------------


def _analyze_run(
    run_dir: Path, base_cfg: HeadroomConfig, real_csvs: dict[str, Path], m0_template: Path
) -> dict[str, Any]:
    """One run's record; any failure becomes {"run", "error"} and nothing else."""
    record: dict[str, Any] = {"run": run_dir.name, "error": None}
    try:
        record.update(_analyze_run_inner(run_dir, base_cfg, real_csvs, m0_template))
    except Exception as exc:  # one broken run must not sink the whole report
        record["error"] = str(exc)
    return record


def _analyze_run_inner(
    run_dir: Path, base_cfg: HeadroomConfig, real_csvs: dict[str, Path], m0_template: Path
) -> dict[str, Any]:
    replay = _Replay(run_dir, base_cfg, real_csvs, m0_template)
    cfg = replay.cfg
    rounds = _read_jsonl(run_dir / "rounds.jsonl")
    arm = next((str(row["arm"]) for row in rounds if row.get("arm")), None)
    # The incumbent chain: accepted rounds in order, m0 first.  Ids cannot
    # repeat in a healthy ledger; the membership guard is cheap insurance.
    chain = ["m0"]
    for row in rounds:
        if row.get("accepted") and row["candidateId"] not in chain:
            chain.append(str(row["candidateId"]))
    accepted_rows = [row for row in rounds if row.get("accepted")]
    evaluated_rows = [row for row in rounds if row.get("status") == "evaluated"]

    # ---- incumbent chain replay on the test segment --------------------------------
    start = cfg.input_length - 1
    usable = segment_origins(start, replay.test_values.shape[0], cfg)
    _, y_test = windows(replay.test_values, usable, cfg)
    if replay.oracle is None:
        mse_oracle = None
    else:
        oracle_pred = np.asarray(replay.oracle.forecast(replay.test_values, usable, cfg), dtype=np.float64)
        mse_oracle = float(np.mean(np.square(y_test - oracle_pred)))
    incumbents: list[dict[str, Any]] = []
    for position, cand_id in enumerate(chain, start=1):
        state = replay.test_state(cand_id)
        if state is None:
            raise _RunError(
                f"incumbent {cand_id!r} could not be replayed: {replay.errors.get(cand_id, 'unknown error')}"
            )
        if not np.array_equal(state["seg_test"].origins, usable):
            raise _RunError("test usable origins differ across incumbents; cannot align scores")
        corrected = baselines.uniform_m0g_postprocess(
            state["pred_test"], state["seg_diag"], state["seg_test"], cfg, state["peaks"]
        )
        mse = float(np.mean(np.square(y_test - corrected)))
        gap = (mse - mse_oracle) / mse_oracle if mse_oracle is not None and mse_oracle > 0.0 else None
        incumbents.append(
            {
                "id": cand_id,
                "position": position,
                "testMse": mse,
                "mseOracle": mse_oracle,
                "gap": gap,
                "normalizedGap": None,
                "mseRatio": None,
            }
        )

    # Normalization: ngap against m0's gap (synthetic; real has no oracle).
    if replay.is_real:
        kind, base_value = "mseRatio", incumbents[0]["testMse"]
    else:
        kind, base_value = "normalizedGap", incumbents[0]["gap"]
    for inc in incumbents:
        value = inc["gap"] if kind == "normalizedGap" else inc["testMse"]
        inc[kind] = (
            value / base_value if value is not None and base_value is not None and base_value > 0.0 else None
        )
    series_values = [inc[kind] for inc in incumbents]  # indexed by chain position - 1
    e1 = series_values[-1]
    curve: list[float | None] = []
    accepted_before = 0
    for row in rounds:  # the incumbent's value after each round index
        if row.get("accepted"):
            accepted_before += 1
        curve.append(series_values[accepted_before])
    e2 = float(np.mean(curve)) if curve and all(v is not None for v in curve) else None

    # ---- run diagnostics ------------------------------------------------------------
    tokens = {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0, "reasoningTokens": 0}
    seconds = 0.0
    for row in rounds:
        for key in tokens:
            tokens[key] += int((row.get("tokens") or {}).get(key, 0))
        seconds += float(row.get("seconds", 0.0))
    types = {"numpy": 0, "torch": 0}
    for row in evaluated_rows:
        path = replay.ws / "models" / "candidates" / str(row.get("candidateId", "")) / "model.py"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            types["torch" if ("import torch" in text or "from torch" in text) else "numpy"] += 1

    # ---- directed / off-target rates (all arms) --------------------------------------
    directed = off_target = compared_evaluated = compared_accepted = 0
    for row in evaluated_rows:
        aspect = (row.get("proposal") or {}).get("targetAspect")
        top = _top_aspect(replay.card(str(row.get("parent"))))
        if aspect is None or top is None:
            continue
        compared_evaluated += 1
        if aspect == top:
            directed += 1
        if row.get("accepted"):
            compared_accepted += 1
            if aspect != top:
                off_target += 1

    # ---- cert-level total LCB vs true test gap (accepted incumbents) ------------------
    # Fixed certification level (the alpha ledger is gone; the loop's
    # certification log is a record, not a budget).
    cert_level = 1.0 - cfg.certify_alpha
    points: list[list[float]] = []
    for inc in incumbents[1:]:  # the spec's "accepted incumbents": m0 excluded
        state = replay.candidate_state(inc["id"])
        fitted = replay.fitted_total(inc["id"])
        if state is None or fitted is None:
            continue
        lcb = float(corrector_gain(fitted, state["seg_cert"], cfg, level=cert_level).lcb)
        inc["certTotalLcb"] = lcb
        if inc["gap"] is not None:
            points.append([lcb, inc["gap"]])
    correlation: float | None = None
    if len(points) >= 3:
        arr = np.asarray(points, dtype=np.float64)
        if arr[:, 0].std() > 0.0 and arr[:, 1].std() > 0.0:
            correlation = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])

    # ---- certification stop point ----------------------------------------------------
    certs = _read_jsonl(replay.heldout / "certification.jsonl")
    stop_point: dict[str, Any] | None = None
    ngap_at_stop: float | None = None
    for index, entry in enumerate(certs, start=1):
        if entry.get("saturated"):  # the round-0 m0 entry counts
            stop_point = {
                "certIndex": index,
                "round": entry.get("round"),
                "incumbentId": entry.get("incumbentId"),
            }
            ngap_at_stop = {inc["id"]: inc[kind] for inc in incumbents}.get(
                str(entry.get("incumbentId"))
            )
            break
    post_stop_gain = (
        abs(ngap_at_stop) - abs(e1)
        if stop_point is not None and ngap_at_stop is not None and e1 is not None
        else None
    )

    # ---- audit -----------------------------------------------------------------------
    violations = _audit_paths(run_dir)
    gain_flags: list[dict[str, Any]] = []
    for row in evaluated_rows:
        cand_id = str(row.get("candidateId", ""))
        state = replay.candidate_state(cand_id)
        fitted = replay.fitted_total(cand_id)
        if state is None or fitted is None:
            continue
        card = replay.card(cand_id)
        diag_gain = next(
            (float(item["delta"]) for item in (card or []) if item.get("label") == "total"), None
        )
        if diag_gain is None or diag_gain <= 0.0:
            continue  # nothing that looked like a gain on diag
        cert_gain = float(corrector_gain(fitted, state["seg_cert"], cfg).delta)
        ratio = diag_gain / cert_gain if cert_gain > 0.0 else None  # cert_gain <= 0: infinite distrust
        if cert_gain <= 0.0 or (ratio is not None and ratio > _GAIN_RATIO_FLAG):
            gain_flags.append(
                {"candidateId": cand_id, "diagGain": diag_gain, "certGain": cert_gain, "ratio": ratio}
            )

    return {
        "task": replay.task_name,
        "arm": arm,
        "seed": replay.seed,
        "real": replay.is_real,
        "rounds": len(rounds),
        "incumbentChain": chain,
        "incumbents": incumbents,
        "e1": e1,
        "e1Kind": kind,
        "e2": e2,
        "acceptances": len(accepted_rows),
        "evaluated": len(evaluated_rows),
        "failureRate": (1.0 - len(evaluated_rows) / len(rounds)) if rounds else None,
        "tokens": tokens,
        "seconds": seconds,
        "candidateTypes": types,
        "directedRate": directed / compared_evaluated if compared_evaluated else None,
        "directedCount": directed,
        "directedDenominator": compared_evaluated,
        "offTargetRate": off_target / compared_accepted if compared_accepted else None,
        "offTargetCount": off_target,
        "offTargetDenominator": compared_accepted,
        "lcbGapCorrelation": correlation,
        "lcbGapPoints": points,
        "stopPoint": stop_point,
        "ngapAtStop": ngap_at_stop,
        "postStopGain": post_stop_gain,
        "evolution": _evolution_accounting(rounds, replay.card),
        "probes": _probe_accounting(run_dir, rounds),
        "audit": {
            "flagged": bool(violations),
            "violations": violations,
            "gainRatioFlags": gain_flags,
            "seriesMatchesTask": replay.series_matches_task,
            "replayErrors": replay.errors or None,
        },
    }


# --- arm-level statistics ---------------------------------------------------------------



def _aspect_lcb(card_rows: list[dict[str, Any]] | None) -> dict[str, float]:
    """Per aspect, the strongest remaining-improvement bound in a posthoc card."""
    strongest: dict[str, float] = {}
    if not card_rows:
        return strongest
    for row in card_rows:
        aspect = str(row.get("aspect"))
        lcb = row.get("lcb")
        value = float("-inf") if lcb is None else float(lcb)
        strongest[aspect] = max(strongest.get(aspect, float("-inf")), value)
    return strongest


def _diagnostic_attribution(
    child_card: list[dict[str, Any]] | None, parent_card: list[dict[str, Any]] | None
) -> tuple[str, ...]:
    """Diagnostic-only direction attribution: the aspect whose remaining-
    improvement bound dropped the most.  The live mechanism is the LLM
    judge; this single rule exists so postSwitchDiversity keeps its
    pre-registered meaning as a measurement."""
    parent_lcb = _aspect_lcb(parent_card)
    child_lcb = _aspect_lcb(child_card)
    drops = {
        aspect: parent_lcb[aspect] - child_lcb.get(aspect, float("-inf"))
        for aspect, value in parent_lcb.items()
        if value > 0.0
    }
    if not drops:
        return ()
    best = max(drops, key=lambda aspect: drops[aspect])
    return (best,) if drops[best] > 0.0 else ()


def _evolution_accounting(
    rounds: list[dict[str, Any]], card_of: "Callable[[str], list[dict[str, Any]] | None]"
) -> dict[str, Any]:
    """Spec v2 diagnostics: switchCount and postSwitchDiversity.

    The diversity switch recommends a parent with a residual signature
    disjoint from the incumbent line's; the diagnostic checks whether the
    first ACCEPTANCE after a switch actually closed a disjoint family.
    """
    by_id = {str(row.get("candidateId")): row for row in rounds if row.get("candidateId")}
    accepted = [row for row in rounds if row.get("accepted")]

    def signature_of(candidate_id: str) -> tuple[str, ...]:
        row = by_id.get(candidate_id)
        if row is None or candidate_id == "m0":
            return ()
        return _diagnostic_attribution(card_of(candidate_id), card_of(str(row.get("parent", "m0"))))

    switches: list[dict[str, Any]] = []
    for row in rounds:
        if not row.get("inspirationParent"):
            continue
        reference = signature_of(str(row.get("parent", "m0")))
        after = next((a for a in accepted if int(a["round"]) > int(row["round"])), None)
        entry: dict[str, Any] = {
            "round": int(row["round"]),
            "inspirationParent": row["inspirationParent"],
            "preSwitchSignature": list(reference),
        }
        if after is None:
            entry["firstAcceptedAfter"] = None
            entry["disjoint"] = None
        else:
            sig_after = signature_of(str(after["candidateId"]))
            entry["firstAcceptedAfter"] = after["candidateId"]
            entry["postSwitchSignature"] = list(sig_after)
            entry["disjoint"] = bool(sig_after) and set(sig_after).isdisjoint(reference)
        switches.append(entry)
    return {
        "switchCount": len(switches),
        "switches": switches,
    }


def _probe_accounting(run_dir: Path, rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Spec v2 probeAccounting: written/validated/kept, render reach, wall-clock."""
    validation_path = run_dir / "ws" / "probes" / "validation.jsonl"
    if not validation_path.is_file():
        return None
    validations = _read_jsonl(validation_path)
    library_path = run_dir / "ws" / "probes" / "library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.is_file() else []
    return {
        "written": len(validations),
        "validated": sum(1 for row in validations if row.get("ok")),
        "kept": len(library),
        "roundsWithProbeOutput": (
            sum(1 for row in rounds if int(row.get("probeOutputsRendered", 0) > 0) / len(rounds))
            if rounds
            else None
        ),
        "probesSeconds": sum(float(row.get("probesSeconds", 0.0)) for row in rounds),
    }


def _stats(values: list[float]) -> dict[str, Any]:
    """Median + IQR of one endpoint column (n counts the non-null values)."""
    if not values:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, median, q75 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    return {"n": len(values), "median": median, "q25": q25, "q75": q75, "iqr": q75 - q25}


def _endpoint_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(task, arm) endpoint summary: median + IQR of E1 and E2."""
    groups: dict[tuple[str, str], dict[str, list[Any]]] = {}
    for rec in records:
        task, arm = rec.get("task"), rec.get("arm")
        if rec.get("error") is not None or task is None or arm is None:
            continue
        bucket = groups.setdefault((str(task), str(arm)), {"e1": [], "e2": []})
        for key in ("e1", "e2"):
            bucket[key].append(rec.get(key))
    rows = []
    for (task, arm) in sorted(groups):
        bucket = groups[(task, arm)]
        rows.append(
            {
                "task": task,
                "arm": arm,
                "nRuns": len(bucket["e1"]),
                "e1": _stats([float(v) for v in bucket["e1"] if v is not None]),
                "e2": _stats([float(v) for v in bucket["e2"] if v is not None]),
            }
        )
    return rows


def _perm_contrast(diffs: list[float]) -> dict[str, Any]:
    """Sign-flip permutation within (task, seed) blocks on the mean difference.

    Exact up to _EXACT_PERM_CAP patterns; Monte Carlo beyond.  The observed
    identity flip is part of the exact enumeration, and the Monte Carlo
    branch adds one hypothetical draw at the observed value, so p is never
    0.  The fixed MC seed keeps the reported p reproducible.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n_blocks = int(d.size)
    observed = float(d.mean())
    if 2**n_blocks <= _EXACT_PERM_CAP:
        flips = np.array(list(itertools.product((-1.0, 1.0), repeat=n_blocks)))
        stats = flips @ d / n_blocks
        mode = "exact"
    else:
        rng = np.random.default_rng(0)
        flips = rng.choice((-1.0, 1.0), size=(_MC_PERMS, n_blocks))
        stats = flips @ d / n_blocks
        mode = "monte-carlo"
    exceed = int(np.count_nonzero(np.abs(stats) >= abs(observed) - 1e-12))
    p_value = exceed / stats.size if mode == "exact" else (exceed + 1) / (stats.size + 1)
    low, high = (float(v) for v in np.percentile(stats, [2.5, 97.5]))
    return {
        "statistic": observed,
        "pValue": float(p_value),
        "permLow": low,
        "permHigh": high,
        "permutations": mode,
    }


def _contrast(records: list[dict[str, Any]], arm_a: str, arm_b: str) -> dict[str, Any]:
    """Seed-blocked permutation contrast of E1 between two arms.

    (task, seed) is the block: the three arms share one seed set per task,
    so sign-flipping within a block is the pre-registered randomization.
    Duplicate cells (if a grid was ever re-run) collapse to their mean E1
    first so no seed votes twice.
    """
    cells: dict[tuple[str, int, str], list[float]] = {}
    for rec in records:
        if rec.get("error") is not None or rec.get("arm") not in (arm_a, arm_b):
            continue
        if rec.get("e1") is None:
            continue
        key = (str(rec.get("task")), int(rec.get("seed")), str(rec["arm"]))
        cells.setdefault(key, []).append(float(rec["e1"]))
    blocks: dict[tuple[str, int], dict[str, float]] = {}
    for (task, seed, arm), runs in cells.items():
        blocks.setdefault((task, seed), {})[arm] = float(np.mean(runs))
    diffs = [
        pair[arm_a] - pair[arm_b]
        for pair in blocks.values()
        if arm_a in pair and arm_b in pair
    ]
    result: dict[str, Any] = {"contrast": f"{arm_a}-vs-{arm_b}", "nBlocks": len(diffs)}
    if not diffs:
        result.update(
            {"statistic": None, "pValue": None, "permLow": None, "permHigh": None, "permutations": None}
        )
        return result
    result.update(_perm_contrast(diffs))
    return result


# --- entry points ------------------------------------------------------------------------


def _discover_runs(runs_dir: Path) -> list[Path]:
    """Subdirectories carrying a rounds ledger (or the directory itself)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise _RunError(f"runs directory does not exist: {runs_dir}")
    if (runs_dir / "rounds.jsonl").is_file():
        return [runs_dir]
    return sorted(path for path in runs_dir.iterdir() if path.is_dir() and (path / "rounds.jsonl").is_file())


def analyze_runs(
    runs_dir: Path,
    config: Path | None = None,
    real_csvs: dict[str, Path] | None = None,
    m0_template: str = _DEFAULT_M0_TEMPLATE,
) -> dict[str, Any]:
    """Analyze every run under ``runs_dir``; returns a JSON-ready dict.

    Every run is independent: a broken run yields {"run", "error"} and the
    table, contrasts and audit summaries simply skip it.
    """
    base_cfg = HeadroomConfig.from_json(config) if config is not None else HeadroomConfig()
    template_path = _REPO_ROOT / "experiments" / "templates" / m0_template
    if not template_path.is_file():
        raise _RunError(f"m0 template does not exist: {template_path}")
    records = [
        _analyze_run(run_dir, base_cfg, real_csvs or {}, template_path)
        for run_dir in _discover_runs(Path(runs_dir))
    ]
    flagged = [rec["run"] for rec in records if rec.get("audit", {}).get("flagged")]
    gain_flagged = [rec["run"] for rec in records if rec.get("audit", {}).get("gainRatioFlags")]
    return _json_safe(
        {
            "runsDir": str(runs_dir),
            "config": str(config) if config is not None else None,
            "runs": records,
            "endpointTable": _endpoint_table(records),
            "contrasts": [_contrast(records, arm_a, arm_b) for arm_a, arm_b in _CONTRASTS],
            "audit": {"pathViolationRuns": flagged, "gainRatioFlagRuns": gain_flagged},
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="directory of loop run directories")
    parser.add_argument("--out", type=Path, default=None, help="output JSON (default: <runs>/analysis.json)")
    parser.add_argument(
        "--config", type=Path, default=None, help="HeadroomConfig JSON for the replay (default: frozen defaults)"
    )
    parser.add_argument(
        "--m0-template",
        default=_DEFAULT_M0_TEMPLATE,
        help=f"m0 template filename under experiments/templates (default: {_DEFAULT_M0_TEMPLATE})",
    )
    parser.add_argument(
        "--real-csv",
        action="append",
        default=[],
        metavar="TASK=CSV",
        help="map a real run's task name to its CSV path (repeatable)",
    )
    args = parser.parse_args(argv)
    real_csvs: dict[str, Path] = {}
    for item in args.real_csv:
        name, sep, path = item.partition("=")
        if not (sep and name and path):
            parser.error(f"--real-csv expects TASK=CSV, got {item!r}")
        real_csvs[name] = Path(path)
    if not args.runs.is_dir() or not _discover_runs(args.runs):
        parser.error(f"no run directories under {args.runs}")
    report = analyze_runs(
        args.runs, config=args.config, real_csvs=real_csvs, m0_template=args.m0_template
    )
    out = args.out if args.out is not None else Path(args.runs) / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out} ({len(report['runs'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
