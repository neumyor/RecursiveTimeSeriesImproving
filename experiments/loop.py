from __future__ import annotations

"""The standalone LLM candidate-improvement loop (final plan §3, rev. §12).

One run = one (task, arm, seed) cell.  Each round opens a fresh DSH session
whose only tools are read/glob/grep/write/edit (no bash), renders the round
prompt from the shared templates plus the arm-specific feedback block, and
lets the agent write exactly one candidate.  Evaluation, acceptance (DM +
HAC), and epsilon-saturation certification run OUTSIDE the agent; the cert
segment and every certification number live in a heldout directory the agent
never sees.  Checkpointing is per-round: rounds.jsonl is the resume ledger.

Post-revision additions (plan §12, 2026-10-16):
- the default incumbent is the PatchTST-lite template (deep m0), so the
  self-built probes can analyze transformer activations;
- the ``self-built`` arm lets the agent write probes under probes/; new
  probes are validated on planted-signal fixtures (experiments/probes.py)
  and the passers are re-run on the incumbent every round — that output is
  the arm's ONLY diagnostic feedback;
- residual-signature evolution guidance (cfg.evolution, on in every arm):
  when the certificate saturates or the last accepted fixes concentrate on
  one residual-structure family, the round prompt recommends a different
  archive candidate (best evaluated candidate with a disjoint signature) as
  the starting point instead of the incumbent.

Design notes from the M2 review (all load-bearing):
- the ledger tolerates a torn last line (a kill mid-append must not kill the
  resume), records the failure reason of every round, and never emits bare
  NaN (strict-JSON readers would choke);
- the acceptance transaction is idempotent under crashes: certification j is
  derived from the certification log (append-only), and resume reconciles
  models/incumbent against the ledger;
- the placebo permutation is recomputed per round from (seed, row count) —
  the card's row count changes with the incumbent's period peaks;
- accepted candidates are determinism-checked (subprocess vs in-process
  predictions must agree) before the incumbent is replaced.

Usage:
    uv run --extra research python experiments/loop.py \
        --run-dir experiments/runs/pilot --task hmm_ar_switch --arm fixed
"""

import argparse
import asyncio
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import evaluate as ev
from harnessing_ts.agent.sdk_runner import SdkRunner, SdkRunnerConfig
from harnessing_ts.headroom import candidate as cand
from harnessing_ts.headroom.certify import certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    _min_bandwidth,
    all_gains,
    card_level,
    select_period_peaks,
)
from harnessing_ts.headroom.feedback import render_feedback
from harnessing_ts.headroom.inference import auto_bandwidth, dm_one_sided_p
from harnessing_ts.headroom.residuals import SegmentData
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.usage import workspace_usage
from harnessing_ts.settings.llm import build_sdk_invocation_config, read_effective_llm_config
from harnessing_ts.state.message_log import MessageLog

AGENT_TOOLS = ["read", "glob", "grep", "write", "edit"]
HELDOUT_ROOT = Path.home() / ".rhg-heldout"
MAX_REPAIRS = 2
SEND_RETRIES = 2  # extra attempts after a transient DSH runtime failure
PROPOSAL_KEYS = ("candidateId", "targetAspect", "aspectDetail", "mechanism", "expectation")
PROBE_PROPOSAL_KEYS = (
    "probeId",
    "measures",
    "signalKey",
    "expectedDirection",
    "rationale",
    "kind",  # "residual" | "activation": which artifact family the probe reads
)
ASPECT_VOCABULARY = (
    "periodicity",
    "lag-structure",
    "long-range-beyond-window",
    "cross-channel",
    "horizon-range",
    "level-bias",
    "other",
)
# Diverse-parent switch (plan §12.3): fires on certificate saturation or
# INSPIRATION_ASPECT_RUN consecutive acceptances closing the same family.
INSPIRATION_ASPECT_RUN = 3
# Activation sample size handed to probes (plan §12.2): evenly spaced
# diagnostic windows, per-layer, of the current incumbent.
ACTIVATION_WINDOWS = 256


class RoundRunner(Protocol):
    """What the loop needs from an LLM session (mockable in tests)."""

    async def send(self, prompt: str) -> list[Any]: ...
    async def close(self) -> None: ...


@dataclass
class LoopState:
    """Everything a resume needs, reconstructed from rounds.jsonl + workspace."""

    round_index: int  # next round to run (1-based)
    incumbent_id: str
    history: list[dict[str, Any]]


def _json_safe(value: Any) -> Any:
    """NaN -> None recursively; numpy scalars -> plain Python for strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant read: a torn LAST line (kill mid-append) is dropped."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                return records  # torn tail from an interrupted append
            raise ValueError(f"{path} is corrupted at line {index + 1}") from None
    return records


def _load_module(name: str, source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _file_digest(path: Path) -> str:
    """Short SHA-256 for the two integrity-critical files (targeted, not a
    tree hash: the §0.5 ban was on hashing the whole workspace)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hgroup_mse(seg: SegmentData) -> dict[str, float]:
    half = seg.r.shape[1] // 2
    return {
        f"hgroup[1,{half}]": float(np.mean(np.square(seg.r[:, :half, :]))),
        f"hgroup[{half + 1},{seg.r.shape[1]}]": float(np.mean(np.square(seg.r[half:, :, :]))),
    }


def _pin_torch_threads() -> None:
    """One BLAS thread once a candidate imports torch: the determinism gate
    compares in-process predictions against the subprocess's, and threaded
    BLAS may reduce in a different order at different thread counts."""
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


class HeadroomLoop:
    def __init__(
        self,
        *,
        run_dir: Path,
        task_name: str,
        arm: str,
        seed: int,
        cfg: HeadroomConfig,
        sizes: dict[str, int] | None = None,
        real_csv: Path | None = None,
        runner_factory: Callable[[Path, str, int], RoundRunner] | None = None,
        m0_template: str = "m0_deep.py",
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.ws = self.run_dir / "ws"
        self.heldout: Path | None = None  # resolved in prepare()
        self.task_name = task_name
        self.arm = arm
        self.seed = seed
        self.cfg = replace(cfg, seed=seed)
        self.sizes = sizes or {"train": 20000, "diag": 10000, "cert": 30000, "test": 52000}
        self.real_csv = real_csv
        self.runner_factory = runner_factory or self._dsh_runner_factory
        # The revision's deep default (plan §12.2); tests use the linear
        # template to stay fast, production always runs the PatchTST-lite.
        self.m0_template = m0_template
        self.rounds_path = self.run_dir / "rounds.jsonl"
        self.train: np.ndarray | None = None
        self.values: np.ndarray | None = None  # train+diag contiguous
        self.cert_values: np.ndarray | None = None
        self.diag_start = 0
        self.seg_diag: SegmentData | None = None
        self.seg_cert: SegmentData | None = None
        self.period_peaks: list[int] = []
        self._segments_incumbent_id: str | None = None
        # Self-built arm state: the in-memory activation sample of the current
        # incumbent (never written into ws — probes receive it as an input),
        # and the probe-output cache keyed by (incumbent, library size).
        self._activations: dict[str, np.ndarray] | None = None
        self._activation_origins: np.ndarray | None = None
        self._incumbent_model_id: str = "m0"
        self._probe_cache: tuple[tuple[str, int], list[dict[str, Any]]] | None = None
        self._probe_seconds = 0.0
        self._last_probe_output_count = 0
        self._probes_module: Any | None = None

    # ---------------------------------------------------------------- setup

    def prepare(self) -> None:
        """Create the workspace, the m0 incumbent, and the heldout records."""
        fresh = not self.rounds_path.is_file()
        if fresh:
            # A random heldout subdir: re-running a cell must never append to
            # a previous run's certification log (that would corrupt the
            # post-hoc stop-point analysis).  The path is persisted so a
            # resume keeps writing to the same place.
            self.heldout = HELDOUT_ROOT / f"{uuid4().hex[:8]}-{self.run_dir.name}-{self.arm}-{self.seed}"
            _write_json(self.run_dir / "heldout.json", {"path": str(self.heldout)})
        else:
            self.heldout = Path(json.loads((self.run_dir / "heldout.json").read_text(encoding="utf-8"))["path"])

        if self.real_csv is not None:
            realdata = _load_module("rhg_realdata", _REPO_ROOT / "experiments" / "realdata.py")
            task = realdata.load_real_task(self.real_csv, self.cfg)
            self.values, self.cert_values, self.diag_start = task.train_diag, task.cert, task.diag_start
        else:
            task = REGISTRY[self.task_name](seed=self.seed, sizes=self.sizes)
            self.values, self.cert_values = task.train_diag, task.cert
            self.diag_start = task.diag_start
        self.train = self.values[: self.diag_start]

        self.ws.mkdir(parents=True, exist_ok=True)
        (self.ws / "models" / "candidates").mkdir(parents=True, exist_ok=True)
        (self.ws / "logs").mkdir(parents=True, exist_ok=True)
        (self.ws / "data").mkdir(parents=True, exist_ok=True)
        if self.arm == "self-built":
            (self.ws / "probes").mkdir(parents=True, exist_ok=True)
        self.heldout.mkdir(parents=True, exist_ok=True)
        self.heldout.chmod(0o700)

        np.save(self.ws / "data" / "series.npy", self.values)
        # The agent's write/edit tools can reach these files; the reference
        # digest lives where the agent cannot and every round verifies it.
        _write_json(
            self.heldout / "reference.json",
            {
                "series_sha": _file_digest(self.ws / "data" / "series.npy"),
                "task": self.task_name,
                "sizes": self.sizes,
                "seed": self.seed,
            },
        )
        _write_json(
            self.ws / "meta.json",
            {
                "L": self.cfg.input_length,
                "H": self.cfg.horizon,
                "C": int(self.values.shape[1]),
                "seed": self.seed,
                # The process-family name is the answer for synthetic tasks;
                # the agent-visible workspace only ever says "synthetic".
                "task": "real-deidentified" if self.real_csv is not None else "synthetic",
            },
        )
        incumbent = self.ws / "models" / "incumbent" / "model.py"
        if not incumbent.is_file():
            incumbent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_REPO_ROOT / "experiments" / "templates" / self.m0_template, incumbent)
            _write_json(
                self.heldout / "incumbent-reference.json", {"sha": _file_digest(incumbent)}
            )

    # ------------------------------------------------------- incumbent state

    def _incumbent_source(self, incumbent_id: str) -> Path:
        if incumbent_id == "m0":
            return self.ws / "models" / "incumbent" / "model.py"
        return self.ws / "models" / "candidates" / incumbent_id / "model.py"

    def _refresh_incumbent_segments(self) -> None:
        """Re-predict the workspace incumbent on diag/cert, rebuild segments."""
        assert self.train is not None and self.values is not None and self.cert_values is not None
        module = _load_module("rhg_incumbent", self.ws / "models" / "incumbent" / "model.py")
        adapter = module.create_adapter()
        _pin_torch_threads()  # after the factory: a candidate importing torch inside create_adapter escapes a pre-factory pin
        self._incumbent_model_id = str(getattr(adapter, "model_id", "m0"))
        adapter.fit(self.train, ev.fit_meta(self.cfg, self.values.shape[1], self.seed))
        pred_diag = ev.predict_adapter(adapter, self.values, self.diag_start, self.values.shape[0], self.cfg)
        self.seg_diag = ev.build_segment(self.values, self.diag_start, self.values.shape[0], pred_diag, self.cfg)
        pred_cert = ev.predict_adapter(
            adapter, self.cert_values, self.cfg.input_length - 1, self.cert_values.shape[0], self.cfg
        )
        self.seg_cert = ev.build_segment(
            self.cert_values, self.cfg.input_length - 1, self.cert_values.shape[0], pred_cert, self.cfg
        )
        seg_a, _ = ev.split_ab(self.seg_diag, self.cfg)
        self.period_peaks = select_period_peaks(seg_a.e_full, seg_a.origins, self.cfg)
        self._sample_incumbent_activations(adapter)

    def _sample_incumbent_activations(self, adapter: Any) -> None:
        """Sample per-layer activations of the incumbent for agent probes (§12.2).

        The activations live in memory only: probes receive them as an input
        artifact, the agent never gets a file to read.  ``activations[i]``
        corresponds to residual row ``positions[i]`` of the diagnostic
        segment (the positions recorded in meta["activationOrigins"]), so
        probes can relate hidden state to realized error.  Adapters without
        the ``return_layers`` extension (e.g. a hand-written linear fallback)
        simply yield an empty activation dict.
        """
        assert self.seg_diag is not None and self.values is not None
        self._activations = None
        self._activation_origins = None
        if self.arm != "self-built" or self.seg_diag.origins.size == 0:
            return
        positions = np.linspace(
            0, self.seg_diag.origins.size - 1, min(ACTIVATION_WINDOWS, self.seg_diag.origins.size)
        ).astype(np.int64)
        sample_origins = self.seg_diag.origins[positions]
        x, _ = ev.windows(self.values, sample_origins, self.cfg)
        try:
            result = adapter.predict_batch(x, return_layers=True)
        except TypeError:
            return  # no activation path in this incumbent
        if not (isinstance(result, tuple) and len(result) == 2):
            return
        _, info = result
        try:
            layers = info.get("layers", []) if isinstance(info, dict) else []
            self._activations = {str(name): np.asarray(arr) for name, arr in layers}
        except Exception:  # a garbage layer dump degrades to activation-less
            return
        if not self._activations:
            return
        self._activation_origins = positions

    def _integrity_check(self, incumbent_id: str) -> None:
        """Series and incumbent must be unmodified since prepare/acceptance.

        Targeted, not a tree hash: these two files define the experiment's
        ground truth, and a tampered incumbent would let a trivial edit pass
        the DM acceptance forever.
        """
        assert self.heldout is not None
        reference = json.loads((self.heldout / "reference.json").read_text(encoding="utf-8"))
        if _file_digest(self.ws / "data" / "series.npy") != reference["series_sha"]:
            raise RuntimeError("workspace integrity violation: data/series.npy was modified")
        incumbent = self.ws / "models" / "incumbent" / "model.py"
        expected = self._incumbent_source(incumbent_id)
        if incumbent.read_bytes() != expected.read_bytes():
            raise RuntimeError(
                "workspace integrity violation: models/incumbent/model.py differs from the "
                f"ledger incumbent {incumbent_id}; restore it before resuming"
            )

    # ------------------------------------------------------------- prompts

    def _feedback_gains(self) -> list[Any]:
        """The incumbent's card gains (card_level), empty for the other arms."""
        assert self.seg_diag is not None
        if self.arm not in {"fixed", "placebo"}:
            return []
        seg_a, seg_b = ev.split_ab(self.seg_diag, self.cfg)
        channels = self.seg_diag.values.shape[1]
        gains = all_gains(
            seg_a, seg_b, self.cfg, self.period_peaks, level=card_level(self.cfg, self.period_peaks, channels)
        )
        # The card keeps the total row (the aggregate bound and the mu2
        # level-bias row); hgroups stay certification-only per the contract.
        return [g for g in gains if g.coordinate.kind != "hgroup"]

    def _render_round(
        self,
        state: LoopState,
        feedback_block: str,
        failure_block: str,
        inspiration: tuple[str, str] | None,
    ) -> str:
        assert self.seg_diag is not None
        diag_mse = float(np.mean(np.square(self.seg_diag.r)))
        history_rows = ["| round | candidateId | accepted | diagMse | status |", "|---|---|---|---|---|"]
        for row in state.history[-10:]:
            diag_mse_row = "n/a" if row["diagMse"] is None else f"{row['diagMse']:.4f}"
            history_rows.append(
                f"| {row['round']} | {row['candidateId']} | {row['accepted']} | {diag_mse_row} | {row['status']} |"
            )
        template = (_REPO_ROOT / "experiments" / "prompts" / "round.md").read_text(encoding="utf-8")
        return template.format(
            round=state.round_index,
            budget=self.cfg.budget_candidates,
            diag_mse=diag_mse,
            model_id=state.incumbent_id,
            feedback_block=feedback_block,
            history_block="\n".join(history_rows),
            parent_block=self._parent_block(state, inspiration),
            failure_block=failure_block,
        )

    def _render_feedback(self, state: LoopState, gains: list[Any]) -> str:
        assert self.seg_diag is not None
        incumbent = {
            "diagMse": float(np.mean(np.square(self.seg_diag.r))),
            "mseByHGroup": _hgroup_mse(self.seg_diag),
            "round": state.round_index,
            "budgetLeft": self.cfg.budget_candidates - state.round_index + 1,
        }
        if self.arm == "self-built":
            probe_rows, validation_rows, self._probe_seconds = self._probe_feedback(state)
            body = render_feedback(
                self.arm,
                incumbent,
                gains,
                state.history,
                self.cfg,
                probe_rows=probe_rows,
                validation_rows=validation_rows,
            )
            # The how-to-write-probes guide rides below the dynamic block:
            # scaffold and probe outputs first (problem), contract text after.
            guide = (_REPO_ROOT / "experiments" / "prompts" / "probes.md").read_text(encoding="utf-8")
            return body.rstrip() + "\n\n" + guide.rstrip() + "\n"
        placebo_perm = None
        if self.arm == "placebo":
            # Recomputed every round: the card's row count changes with the
            # incumbent's period peaks, and a cached permutation would crash
            # the arm the moment an accepted fix changes the peak count.
            rng = np.random.default_rng(self.seed * 100_003 + len(gains))
            placebo_perm = tuple(int(i) for i in rng.permutation(max(len(gains), 1)))
        return render_feedback(self.arm, incumbent, gains, state.history, self.cfg, placebo_perm)

    # --------------------------------------------------- self-built probes

    def _probe_infra(self) -> Any:
        """experiments/probes.py, loaded by path like the loop itself."""
        if self._probes_module is None:
            self._probes_module = _load_module("rhg_probes", _REPO_ROOT / "experiments" / "probes.py")
        return self._probes_module

    def _probe_library(self) -> list[dict[str, Any]]:
        """Validated probes: {probeId, kind, measures, signalKey, round}."""
        path = self.ws / "probes" / "library.json"
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _probe_validation_rows(self, round_index: int) -> list[dict[str, Any]]:
        """The previous round's validation verdicts, as feedback rows."""
        records = _read_jsonl(self.ws / "probes" / "validation.jsonl")
        return [
            {"probeId": row["probeId"], "ok": row["ok"], "detail": row["detail"]}
            for row in records
            if row["round"] == round_index - 1
        ]

    def _probe_artifacts(self) -> dict[str, Any]:
        """What a probe receives: the incumbent's residuals and activations.

        No raw series values, no targets — the two channels the paper's story
        names (residual analysis + hidden-layer analysis) are the whole input.
        ``activations[i]`` pairs with residual row ``meta["activationOrigins"][i]``.
        """
        assert self.seg_diag is not None
        return {
            "residuals": {
                "e": self.seg_diag.e_full,
                "r": self.seg_diag.r,
                "origins": self.seg_diag.origins,
            },
            "activations": self._activations or {},
            "meta": {
                "L": self.cfg.input_length,
                "H": self.cfg.horizon,
                "C": int(self.seg_diag.values.shape[1]),
                "T": int(self.values.shape[0]),
                "seed": self.seed,
                "incumbent": self._incumbent_model_id,
                "task": "real-deidentified" if self.real_csv is not None else "synthetic",
                "activationOrigins": (
                    self._activation_origins.tolist()
                    if self._activation_origins is not None
                    else []
                ),
            },
        }

    def _run_library_probes(self) -> list[dict[str, Any]]:
        """Run every validated probe on the current incumbent artifacts."""
        infra = self._probe_infra()
        artifacts = self._probe_artifacts()
        rows: list[dict[str, Any]] = []
        for entry in self._probe_library():
            source = self.ws / "probes" / entry["probeId"] / "probe.py"
            row: dict[str, Any] = {"probeId": entry["probeId"], "measures": entry["measures"]}
            if "sourceSha" in entry and source.is_file():
                if _file_digest(source) != entry["sourceSha"]:
                    # Validation froze this probe's source; an edit after
                    # validation must not run unvalidated.  The agent
                    # resubmits under a new id.
                    row["error"] = (
                        "probe source changed after validation; re-submit it under a new probe id"
                    )
                    rows.append(row)
                    continue
            try:
                row["output"] = infra.run_probe(source, artifacts)
            except Exception as exc:  # a broken probe loses its row, not the round
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        return rows

    def _probe_feedback(self, state: LoopState) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        """(probe output rows, validation rows); outputs cached per (incumbent, library).

        Cache-safety rests on the library being append-only (a re-validated
        probe always takes a new id), so its length is a sufficient identity;
        an acceptance changes the incumbent key and invalidates the cache.
        """
        key = (state.incumbent_id, len(self._probe_library()))
        if self._probe_cache is None or self._probe_cache[0] != key:
            started = time.monotonic()
            self._probe_cache = (key, self._run_library_probes())
            self._probe_seconds = time.monotonic() - started
        else:
            self._probe_seconds = 0.0  # cached: this round spent nothing on runs
        rows = self._probe_cache[1]
        self._last_probe_output_count = len(rows)
        return rows, self._probe_validation_rows(state.round_index), self._probe_seconds

    def _read_probe_proposal(self, probe_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
        """Structural check of one probe proposal; problems become verdicts."""
        source = probe_dir / "probe.py"
        proposal_path = probe_dir / "probe_proposal.json"
        if not source.is_file() or not proposal_path.is_file():
            return None, "probe is missing probe.py or probe_proposal.json"
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"probe_proposal.json is not valid JSON: {exc}"
        if not isinstance(proposal, dict):
            return None, "probe_proposal.json must be a JSON object"
        missing = [key for key in PROBE_PROPOSAL_KEYS if key not in proposal]
        if missing:
            return None, f"probe_proposal.json is missing fields: {', '.join(missing)}"
        if proposal["probeId"] != probe_dir.name:
            return None, "probeId must equal the probe directory name"
        if proposal["kind"] not in ("residual", "activation"):
            return None, 'kind must be "residual" or "activation"'
        if proposal["expectedDirection"] not in ("higher", "lower"):
            return None, 'expectedDirection must be "higher" or "lower"'
        return proposal, None

    def _collect_probes(self, state: LoopState) -> tuple[int, int]:
        """Validate newly written probes on planted fixtures; keep the passers.

        Returns (submitted, kept): how many NEW probes were seen and how many
        passed validation.  The verdict reaches the agent in the NEXT round's
        feedback; a rejected probe stays on disk — the agent revises it under
        a new id (ids are never reused, like candidates).
        """
        probes_dir = self.ws / "probes"
        if not probes_dir.is_dir():
            return 0, 0
        infra = self._probe_infra()
        library = self._probe_library()
        seen = {row["probeId"] for row in library}
        seen |= {row["probeId"] for row in _read_jsonl(probes_dir / "validation.jsonl")}
        submitted = 0
        kept = 0
        for probe_dir in sorted(probes_dir.iterdir()):
            if not probe_dir.is_dir() or probe_dir.name in seen:
                continue
            submitted += 1
            record: dict[str, Any] = {
                "probeId": probe_dir.name,
                "round": state.round_index,
                "ok": False,
                "detail": "",
            }
            proposal, problem = self._read_probe_proposal(probe_dir)
            if problem is not None:
                record["detail"] = problem
            else:
                assert proposal is not None
                result = infra.validate_probe(probe_dir / "probe.py", proposal, proposal["kind"])
                record["ok"] = bool(result["ok"])
                record["detail"] = result.get("reason") or "validated"
                if result["ok"]:
                    kept += 1
                    library.append(
                        {
                            "probeId": probe_dir.name,
                            "kind": proposal["kind"],
                            "measures": proposal["measures"],
                            "signalKey": proposal["signalKey"],
                            "sourceSha": _file_digest(probe_dir / "probe.py"),
                            "round": state.round_index,
                        }
                    )
                    _write_json(probes_dir / "library.json", library)
            _append_jsonl(probes_dir / "validation.jsonl", record)
        return submitted, kept

    # ----------------------------------------------------------- LLM runner

    def _dsh_runner_factory(self, ws: Path, system_prompt: str, round_index: int) -> RoundRunner:
        sdk = build_sdk_invocation_config(read_effective_llm_config(ws))
        runner = SdkRunner(
            SdkRunnerConfig(
                cwd=ws,
                system_prompt=system_prompt,
                allowed_tools=AGENT_TOOLS,
                log=MessageLog(ws / "logs" / f"round-{round_index}.jsonl"),
                model=sdk.model,
                provider=sdk.provider,
                max_tokens=sdk.max_tokens,
                env=sdk.env,
                session_role="main",
            )
        )
        return _SdkRoundRunner(runner)

    async def _send_with_retries(self, state: LoopState, system_prompt: str, round_text: str) -> None:
        """Send one round prompt, retrying transient DSH runtime failures.

        A fresh session per attempt keeps the ledger simple: the round either
        produces a candidate or it does not; a runtime death costs one send.
        """
        last_error: Exception | None = None
        for _attempt in range(1 + SEND_RETRIES):
            runner = self.runner_factory(self.ws, system_prompt, state.round_index)
            try:
                await runner.send(round_text)
                return
            except Exception as exc:  # transient runtime failure; retry
                last_error = exc
            finally:
                await runner.close()
        raise RuntimeError(
            f"round {state.round_index}: DSH runtime failed {1 + SEND_RETRIES} times"
        ) from last_error

    # --------------------------------------------------------------- state

    def load_state(self) -> LoopState:
        rounds = _read_jsonl(self.rounds_path)
        accepted = [r for r in rounds if r.get("accepted")]
        incumbent_id = str(accepted[-1]["candidateId"]) if accepted else "m0"
        history = [
            {
                "round": r["round"],
                "candidateId": r["candidateId"],
                "accepted": r["accepted"],
                "diagMse": r["diagMse"],
                "status": r["status"],
                "evalNote": r.get("evalNote"),
            }
            for r in rounds
        ]
        # Heal the acceptance crash window: the ledger is authoritative, so
        # models/incumbent must match the ledger's incumbent.
        ledger_source = self._incumbent_source(incumbent_id)
        incumbent = self.ws / "models" / "incumbent" / "model.py"
        if not incumbent.is_file():
            incumbent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ledger_source, incumbent)
        elif incumbent.read_bytes() != ledger_source.read_bytes():
            shutil.copyfile(ledger_source, incumbent)
        if self._segments_incumbent_id != incumbent_id:
            self._refresh_incumbent_segments()
            self._segments_incumbent_id = incumbent_id
        return LoopState(round_index=len(rounds) + 1, incumbent_id=incumbent_id, history=history)

    def _certify_incumbent(self, state: LoopState, incumbent_id: str) -> None:
        """Append one certification of the CURRENT incumbent (heldout only).

        The rule is one sentence: on the cert segment, every statistic's
        realized-gain 95% UCB must sit below epsilon times its group MSE for
        the incumbent to count as saturated.  The log is a record, not a
        budget; there is no cross-version alpha accounting (2026-10-16
        author-directed simplification).
        """
        assert self.heldout is not None and self.seg_diag is not None and self.seg_cert is not None
        level = 1.0 - self.cfg.certify_alpha
        gains = all_gains(self.seg_diag, self.seg_cert, self.cfg, self.period_peaks, level=level)
        mse = ev.mse_by_group(self.seg_cert, gains)
        report = certify(gains, mse, self.cfg)
        _append_jsonl(
            self.heldout / "certification.jsonl",
            {
                "round": state.round_index,
                "incumbentId": incumbent_id,
                "saturated": report.saturated,
                "level": report.level,
                "failures": report.failures,
            },
        )

    def _write_posthoc_card(
        self,
        candidate_id: str,
        round_index: int,
        proposal: dict[str, Any],
        seg: SegmentData,
        diag_mse: float,
        dm_p: float | None,
    ) -> None:
        """Heldout posthoc card of one evaluated model (every arm, every round).

        The evolution archive and the post-hoc chain analysis both read these.
        """
        assert self.heldout is not None
        seg_a, seg_b = ev.split_ab(seg, self.cfg)
        peaks = select_period_peaks(seg_a.e_full, seg_a.origins, self.cfg)
        channels = seg.values.shape[1]
        posthoc = all_gains(seg_a, seg_b, self.cfg, peaks, level=card_level(self.cfg, peaks, channels))
        _write_json(
            self.heldout / "posthoc" / f"{candidate_id}.json",
            {
                "round": round_index,
                "proposal": proposal,
                "diagMse": diag_mse,
                "dmP": dm_p,
                "card": [
                    {
                        "label": g.coordinate.label,
                        "aspect": g.coordinate.aspect,
                        "delta": g.delta,
                        "lcb": g.lcb,
                        "ucb": g.ucb,
                        "mu2": g.mu2,
                        "by_horizon": g.by_horizon,
                    }
                    for g in posthoc
                ],
            },
        )

    # ------------------------------------------------- evolution (plan §12.3)

    def _posthoc_card(self, candidate_id: str) -> list[dict[str, Any]] | None:
        """The heldout posthoc card rows of one evaluated candidate, or None."""
        assert self.heldout is not None
        path = self.heldout / "posthoc" / f"{candidate_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["card"]

    @staticmethod
    def _aspect_lcb(card: list[dict[str, Any]] | None) -> dict[str, float]:
        """Per aspect, the strongest remaining-improvement bound in the card."""
        strongest: dict[str, float] = {}
        if not card:
            return strongest
        for row in card:
            aspect = str(row["aspect"])
            lcb = row.get("lcb")
            value = -np.inf if lcb is None else float(lcb)
            strongest[aspect] = max(strongest.get(aspect, -np.inf), value)
        return strongest

    def _candidate_signature(self, candidate_id: str, parent_id: str) -> tuple[str, ...]:
        """Aspect families this evaluated candidate closed vs its parent.

        Closed = the parent's card showed remaining improvement on an aspect
        and the candidate's does not.  If nothing fully closed, the fallback is
        the single aspect whose bound dropped the most (a real improvement that
        shrank every coordinate without finishing one still has a direction).
        An aspect absent from the candidate's card (e.g. its period peaks
        vanished, so no band rows exist) counts as closed: no detectable
        remaining structure on that aspect.
        """
        parent_lcb = self._aspect_lcb(self._posthoc_card(parent_id))
        child_lcb = self._aspect_lcb(self._posthoc_card(candidate_id))
        closed = {a for a, value in parent_lcb.items() if value > 0.0 and child_lcb.get(a, -np.inf) <= 0.0}
        if closed:
            return tuple(sorted(closed))
        drops = {a: parent_lcb[a] - child_lcb.get(a, -np.inf) for a, value in parent_lcb.items() if value > 0.0}
        if not drops:
            return ()
        best = max(drops, key=lambda aspect: drops[aspect])
        return (best,) if drops[best] > 0.0 else ()

    def _evolution_recommendation(self, state: LoopState) -> tuple[str, str, float] | None:
        """(inspiration parent id, trigger, its diag MSE) when the switch fires.

        Triggers: the latest certification of the incumbent line is saturated,
        or the last INSPIRATION_ASPECT_RUN acceptances all closed the same
        single aspect family.  The recommendation is then the best evaluated
        candidate (lowest diagnostic MSE) whose signature is nonempty and
        disjoint from the incumbent line's latest signature — the diversity
        axis is the residual-coordinate space the certificate measures, which
        is the time-series-specific part of the mechanism.
        """
        if not self.cfg.evolution:
            return None
        assert self.heldout is not None
        rounds = _read_jsonl(self.rounds_path)
        accepted = [row for row in rounds if row.get("accepted")]
        cert = _read_jsonl(self.heldout / "certification.jsonl")
        saturated = bool(cert) and bool(cert[-1].get("saturated"))
        recent = accepted[-INSPIRATION_ASPECT_RUN:]
        recent_signatures = [
            self._candidate_signature(row["candidateId"], row["parent"]) for row in recent
        ]
        concentrated = (
            len(recent_signatures) == INSPIRATION_ASPECT_RUN
            and all(len(signature) == 1 for signature in recent_signatures)
            and len(set(recent_signatures)) == 1
        )
        if not (saturated or concentrated):
            return None
        reference = recent_signatures[-1] if recent_signatures else ()
        best: dict[str, Any] | None = None
        for row in rounds:
            if row.get("status") != "evaluated" or row["candidateId"] == state.incumbent_id:
                continue
            signature = self._candidate_signature(row["candidateId"], row["parent"])
            if not signature or set(signature) & set(reference):
                continue
            if best is None or float(row["diagMse"]) < float(best["diagMse"]):
                best = row
        if best is None:
            return None
        return (
            str(best["candidateId"]),
            "saturated" if saturated and not concentrated else "concentrated",
            float(best["diagMse"]),
        )

    def _parent_block(self, state: LoopState, inspiration: tuple[str, str, float] | None) -> str:
        """The diversity-switch paragraph for round.md; empty when no switch.

        Wording is arm-neutral and number-free: no aspect names, no heldout
        figures — the same sentence in every arm, so the feedback-arm
        contrast stays clean (plan §12.3).
        """
        if inspiration is None:
            return ""
        parent_id, trigger, diag_mse = inspiration
        cause = (
            "The measured remaining-improvement bound of the incumbent line has fallen "
            "below the actionable threshold."
            if trigger == "saturated"
            else (
                f"The last {INSPIRATION_ASPECT_RUN} accepted improvements all targeted the "
                "same kind of residual structure."
            )
        )
        mse_note = f" (its evaluated diagnostic MSE was {diag_mse:.4f})"
        return (
            f"{cause}\n\n"
            "Diversity switch for this round: start your candidate from a different "
            f"direction — copy `models/candidates/{parent_id}/model.py`{mse_note} as your "
            "starting point and modify it, instead of starting from the incumbent. "
            f"Acceptance is still measured against the incumbent `{state.incumbent_id}`; "
            "write your own candidate under models/candidates/<your-id>/ as usual."
        )

    async def run(self) -> Path:
        self.prepare()
        state = self.load_state()
        if not state.history:
            # Round-0 certification of the initial incumbent (plan §3.1):
            # without it, a zero-acceptance run would have no stop point.
            self._certify_incumbent(state, "m0")
            # Its posthoc card too, so every accepted candidate's signature
            # (closed aspects vs parent) has a parent card to diff against.
            assert self.seg_diag is not None
            if not (self.heldout / "posthoc" / "m0.json").is_file():
                self._write_posthoc_card(
                    "m0", 0, {}, self.seg_diag, float(np.mean(np.square(self.seg_diag.r))), None
                )
        while state.round_index <= self.cfg.budget_candidates:
            await self._round(state)
            state = self.load_state()
        _write_json(self.heldout / "done.json", {"rounds": len(state.history), "arm": self.arm})
        return self.rounds_path

    # -------------------------------------------------------------- rounds

    async def _round(self, state: LoopState) -> None:
        assert self.seg_diag is not None and self.train is not None and self.heldout is not None
        self._integrity_check(state.incumbent_id)
        self._last_probe_output_count = 0  # set by _render_feedback for the self-built arm
        gains = self._feedback_gains()
        feedback_block = self._render_feedback(state, gains)
        system_prompt = (_REPO_ROOT / "experiments" / "prompts" / "system.md").read_text(encoding="utf-8")
        inspiration = self._evolution_recommendation(state)
        failure_block = ""
        if state.history and state.history[-1].get("evalNote"):
            # An evaluated candidate rejected for a specific reason (the
            # determinism gate) must teach the agent that reason — otherwise
            # the rejection is indistinguishable from a plain DM miss.
            previous = state.history[-1]
            failure_block = (
                f"Note on your previous candidate {previous['candidateId']}: "
                f"{previous['evalNote']}."
            )
        candidate_id = ""
        candidate_seconds: dict[str, float] = {}
        proposal: dict[str, Any] | None = None
        error: str | None = None
        eval_note: str | None = None
        status = "failed"
        diag_mse: float | None = None
        dm_p: float | None = None
        accepted = False
        attempts = 0
        probes_submitted = 0
        probes_kept = 0
        probes_seconds = 0.0
        round_seen: set[str] = set()
        tokens_before = workspace_usage(self.ws)
        started = time.monotonic()

        for _attempt in range(1 + MAX_REPAIRS):
            attempts += 1
            round_text = self._render_round(state, feedback_block, failure_block, inspiration)
            await self._send_with_retries(state, system_prompt, round_text)
            if self.arm == "self-built":
                # Probes are tools, not deliverables: validation never costs a
                # repair turn, and the verdict rides the next round's feedback.
                started = time.monotonic()
                submitted, kept = self._collect_probes(state)
                probes_seconds += time.monotonic() - started
                probes_submitted += submitted
                probes_kept += kept
            candidate_id, source, proposal, problem = self._collect_candidate(state, round_seen)
            if problem is not None:
                error = problem
                failure_block = problem
                continue
            assert source is not None and proposal is not None
            status, diag_mse, dm_p, accepted, round_error, candidate_seconds = self._evaluate(
                state, candidate_id, source, proposal
            )
            if status == "evaluated":
                error = None
                eval_note = round_error  # None on a plain DM miss
                break
            error = round_error
            failure_block = f"Your candidate failed evaluation: {round_error}."

        tokens = {
            key: workspace_usage(self.ws).get(key, 0) - tokens_before.get(key, 0)
            for key in ("inputTokens", "outputTokens", "cacheReadTokens", "reasoningTokens")
        }
        _append_jsonl(
            self.rounds_path,
            {
                "round": state.round_index,
                "candidateId": candidate_id,
                "parent": state.incumbent_id,
                "arm": self.arm,
                "status": status,
                "diagMse": diag_mse,
                "dmP": dm_p,
                "accepted": accepted,
                "error": error,
                "evalNote": eval_note,
                "proposal": proposal or {},
                "tokens": tokens,
                "feedbackChars": len(feedback_block),
                "attempts": attempts,
                "candidateSeconds": candidate_seconds,
                "inspirationParent": inspiration[0] if inspiration else None,
                "probesSubmitted": probes_submitted,
                "probesKept": probes_kept,
                "probesSeconds": round(probes_seconds + self._probe_seconds, 2),
                "probeOutputsRendered": self._last_probe_output_count,
                "seconds": round(time.monotonic() - started, 1),
            },
        )
        if accepted:
            _append_jsonl(
                self.heldout / "acceptances.jsonl",
                {"round": state.round_index, "candidateId": candidate_id},
            )

    def _collect_candidate(
        self, state: LoopState, round_seen: set[str]
    ) -> tuple[str, Path | None, dict[str, Any] | None, str | None]:
        """Find the round's new candidate; a problem string means repair turn."""
        candidates_dir = self.ws / "models" / "candidates"
        known = {row["candidateId"] for row in state.history} | {"m0", ""} | round_seen
        fresh = [
            path
            for path in candidates_dir.iterdir()
            if path.is_dir() and path.name not in known
        ]
        if not fresh:
            return "", None, None, (
                "You did not write a candidate this round. Write "
                "models/candidates/<id>/model.py and proposal.json."
            )
        candidate_dir = max(fresh, key=lambda path: path.stat().st_mtime)
        candidate_id = candidate_dir.name
        round_seen.add(candidate_id)
        source = candidate_dir / "model.py"
        proposal_path = candidate_dir / "proposal.json"
        if not source.is_file() or not proposal_path.is_file():
            return candidate_id, None, None, f"Candidate {candidate_id} is missing model.py or proposal.json."
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return candidate_id, None, None, f"proposal.json is not valid JSON: {exc}."
        if not isinstance(proposal, dict):
            return candidate_id, None, None, "proposal.json must be a JSON object."
        missing = [key for key in PROPOSAL_KEYS if key not in proposal]
        if missing:
            return candidate_id, None, None, f"proposal.json is missing fields: {', '.join(missing)}."
        if proposal.get("candidateId") != candidate_id:
            return candidate_id, None, None, "proposal.json candidateId must equal the directory name."
        if proposal.get("targetAspect") not in ASPECT_VOCABULARY:
            return candidate_id, None, None, (
                f"targetAspect must be one of {', '.join(ASPECT_VOCABULARY)}."
            )
        try:
            cand.check_source(source.read_text(encoding="utf-8"))
        except cand.SourceRejected as exc:
            return candidate_id, None, None, f"model.py was rejected by the source check: {exc}"
        smoke = cand.smoke_test(
            source, candidate_id, self.train, ev.fit_meta(self.cfg, self.values.shape[1], self.seed), self.cfg
        )
        if not smoke["ok"]:
            return candidate_id, None, None, f"model.py failed its smoke test: {smoke['error']}"
        return candidate_id, source, proposal, None

    def _evaluate(
        self,
        state: LoopState,
        candidate_id: str,
        source: Path,
        proposal: dict[str, Any],
    ) -> tuple[str, float | None, float | None, bool, str | None, dict[str, float]]:
        """Subprocess evaluation + DM acceptance + (on accept) certification."""
        assert self.seg_diag is not None and self.train is not None and self.heldout is not None
        assert self.values is not None and self.cert_values is not None
        import tempfile

        with tempfile.TemporaryDirectory(prefix="rhg-cand-") as tmp:
            npz_in = Path(tmp) / "in.npz"
            npz_out = Path(tmp) / "out.npz"
            # Predictions must cover EVERY target-realized origin (the e/r
            # series needs the full prefix); SegmentData keeps only the
            # usable origins internally.
            origins = ev.target_origins(self.diag_start, self.values.shape[0], self.cfg)
            x_diag, _ = ev.windows(self.values, origins, self.cfg)
            # Interleaved batches: consecutive origins differ by one, so
            # batch id = row % (L+H) keeps same-batch rows L+H apart and no
            # batch row's window can contain another's targets.
            batch_ids = np.arange(origins.size, dtype=np.int32) % (self.cfg.input_length + self.cfg.horizon)
            rng = np.random.default_rng(self.seed + state.round_index)
            single = rng.choice(origins.size, size=min(50, origins.size), replace=False).astype(np.int32)
            cert_origins = ev.target_origins(self.cfg.input_length - 1, self.cert_values.shape[0], self.cfg)
            x_cert, _ = ev.windows(self.cert_values, cert_origins, self.cfg)
            np.savez(
                npz_in,
                train=self.train,
                meta=np.array(json.dumps(ev.fit_meta(self.cfg, self.values.shape[1], self.seed))),
                source_path=np.array(str(source.resolve())),
                x_diag=x_diag,
                batch_ids_diag=batch_ids,
                single_check_idx=single,
                x_cert=x_cert,
            )
            try:
                cand.evaluate_candidate(
                    source, candidate_id, npz_in, npz_out, self.cfg,
                    timeout_seconds=self.cfg.candidate_timeout_seconds,
                )
                result = dict(np.load(npz_out))
            except cand.CandidateTimeout:
                return "failed", None, None, False, f"timed out after {self.cfg.candidate_timeout_seconds:g}s", {}
            except cand.CheatDetected:
                return "failed", None, None, False, (
                    "batch-invariance check failed (predictions depend on the batch)"
                ), {}
            except (cand.CandidateCrash, cand.ContractViolation) as exc:
                return "failed", None, None, False, str(exc), {}

        pred_diag = np.asarray(result["pred_diag"], dtype=np.float64)
        candidate_seconds = {
            "fit": float(result.get("fit_seconds", 0.0)),
            "predict": float(result.get("predict_seconds", 0.0)),
        }
        seg_cand = ev.build_segment(self.values, self.diag_start, self.values.shape[0], pred_diag, self.cfg)
        diag_mse = float(np.mean(np.square(seg_cand.r)))
        d = (np.square(self.seg_diag.r) - np.square(seg_cand.r)).mean(axis=(1, 2))
        dm_p = float(dm_one_sided_p(d, auto_bandwidth(d, _min_bandwidth(self.cfg))))
        accepted = bool(dm_p < self.cfg.acceptance_alpha)

        # Posthoc card for every candidate, every arm (heldout only).
        self._write_posthoc_card(candidate_id, state.round_index, proposal, seg_cand, diag_mse, dm_p)

        if not accepted:
            return "evaluated", diag_mse, dm_p, False, None, candidate_seconds

        # Determinism gate before the incumbent is replaced: the next round's
        # feedback re-fits in-process, and a nondeterministic candidate would
        # make the agent see numbers from a different draw than the ledger.
        # rtol 1e-5 (plan §12.2): a trained network's BLAS reduction order
        # legitimately differs between the subprocess and this process; the
        # threads are pinned so the in-process draw itself is stable.
        module = _load_module("rhg_cand_accept", source)
        adapter = module.create_adapter()
        _pin_torch_threads()
        adapter.fit(self.train, ev.fit_meta(self.cfg, self.values.shape[1], self.seed))
        pred_inproc = ev.predict_adapter(
            adapter, self.values, self.diag_start, self.values.shape[0], self.cfg
        )
        if not np.allclose(pred_inproc, pred_diag, rtol=1e-5, atol=1e-9):
            return "evaluated", diag_mse, dm_p, False, (
                "nondeterministic candidate: in-process re-prediction differs from the "
                "evaluated predictions; seed all randomness from meta['seed']"
            ), candidate_seconds

        shutil.copyfile(source, self.ws / "models" / "incumbent" / "model.py")
        self._refresh_incumbent_segments()
        self._segments_incumbent_id = candidate_id
        self._certify_incumbent(state, candidate_id)
        return "evaluated", diag_mse, dm_p, True, None, candidate_seconds


class _SdkRoundRunner:
    def __init__(self, runner: SdkRunner) -> None:
        self._runner = runner

    async def send(self, prompt: str) -> list[Any]:
        parts = await self._runner.send(prompt)
        if not parts:
            # A turn that produced nothing is a billing/auth failure (the
            # observed failure mode: the endpoint rejects every request and
            # the runtime ends the turn as an empty error turn).  Burning
            # the whole budget as silent no-op rounds is the exact bug that
            # voided the first P2 attempt — classify and raise instead.
            raise RuntimeError(
                "DSH turn returned no parts: the LLM endpoint rejected the request "
                "(check the API account balance and credentials)"
            )
        return parts

    async def close(self) -> None:
        await self._runner.close()


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="hmm_ar_switch")
    parser.add_argument(
        "--arm", default="fixed", choices=["scalar", "fixed", "self-built", "placebo"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--real-csv", type=Path, default=None)
    args = parser.parse_args()
    cfg = HeadroomConfig(arm=args.arm, budget_candidates=args.budget, seed=args.seed)
    loop = HeadroomLoop(
        run_dir=args.run_dir,
        task_name=args.task,
        arm=args.arm,
        seed=args.seed,
        cfg=cfg,
        real_csv=args.real_csv,
    )
    await loop.run()


if __name__ == "__main__":
    asyncio.run(main_async())
