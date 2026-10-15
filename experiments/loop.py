from __future__ import annotations

"""The standalone LLM candidate-improvement loop (final plan §3).

One run = one (task, arm, seed) cell.  Each round opens a fresh DSH session
whose only tools are read/glob/grep/write/edit (no bash), renders the round
prompt from the shared templates plus the arm-specific feedback block, and
lets the agent write exactly one candidate.  Evaluation, acceptance (DM +
HAC), and epsilon-saturation certification run OUTSIDE the agent; the cert
segment and every certification number live in a heldout directory the agent
never sees.  Checkpointing is per-round: rounds.jsonl is the resume ledger.

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
        --run-dir experiments/runs/pilot --task hmm_ar_switch --arm card
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
    certify_level,
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
ASPECT_VOCABULARY = (
    "periodicity",
    "lag-structure",
    "long-range-beyond-window",
    "cross-channel",
    "horizon-range",
    "level-bias",
    "other",
)


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
        self.rounds_path = self.run_dir / "rounds.jsonl"
        self.j_max = cfg.budget_candidates + 1
        self.train: np.ndarray | None = None
        self.values: np.ndarray | None = None  # train+diag contiguous
        self.cert_values: np.ndarray | None = None
        self.diag_start = 0
        self.seg_diag: SegmentData | None = None
        self.seg_cert: SegmentData | None = None
        self.period_peaks: list[int] = []
        self._segments_incumbent_id: str | None = None

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
            shutil.copyfile(_REPO_ROOT / "experiments" / "templates" / "m0.py", incumbent)
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
        """The incumbent's card gains (card_level), empty for scalar/bj."""
        assert self.seg_diag is not None
        if self.arm not in {"card", "placebo"}:
            return []
        seg_a, seg_b = ev.split_ab(self.seg_diag, self.cfg)
        channels = self.seg_diag.values.shape[1]
        gains = all_gains(
            seg_a, seg_b, self.cfg, self.period_peaks, level=card_level(self.cfg, self.period_peaks, channels)
        )
        # The card keeps the total row (the aggregate bound and the mu2
        # level-bias row); hgroups stay certification-only per the contract.
        return [g for g in gains if g.coordinate.kind != "hgroup"]

    def _render_round(self, state: LoopState, feedback_block: str, failure_block: str) -> str:
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
            failure_block=failure_block,
        )

    def _render_feedback(self, state: LoopState, gains: list[Any]) -> str:
        assert self.seg_diag is not None
        placebo_perm = None
        if self.arm == "placebo":
            # Recomputed every round: the card's row count changes with the
            # incumbent's period peaks, and a cached permutation would crash
            # the arm the moment an accepted fix changes the peak count.
            rng = np.random.default_rng(self.seed * 100_003 + len(gains))
            placebo_perm = tuple(int(i) for i in rng.permutation(max(len(gains), 1)))
        incumbent = {
            "diagMse": float(np.mean(np.square(self.seg_diag.r))),
            "mseByHGroup": _hgroup_mse(self.seg_diag),
            "round": state.round_index,
            "budgetLeft": self.cfg.budget_candidates - state.round_index + 1,
            "eFull": self.seg_diag.e_full,
        }
        return render_feedback(self.arm, incumbent, gains, state.history, self.cfg, placebo_perm)

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

    def _certification_count(self) -> int:
        """j for the next certification: the log is the alpha-spending ledger.

        Deriving j from the append-only certification log (not the rounds
        ledger) makes the acceptance transaction idempotent: a crash between
        certification and the rounds append cannot double-spend alpha.
        """
        assert self.heldout is not None
        return len(_read_jsonl(self.heldout / "certification.jsonl"))

    def _certify_incumbent(self, state: LoopState, incumbent_id: str) -> None:
        """Append one certification of the CURRENT incumbent (heldout only)."""
        assert self.heldout is not None and self.seg_diag is not None and self.seg_cert is not None
        j = self._certification_count() + 1
        if j > self.j_max:
            return  # alpha ledger is full; the bound already covers this case
        channels = self.seg_cert.r.shape[2]
        level = certify_level(self.cfg, self.j_max, self.period_peaks, channels)
        gains = all_gains(self.seg_diag, self.seg_cert, self.cfg, self.period_peaks, level=level)
        mse = ev.mse_by_group(self.seg_cert, gains)
        report = certify(gains, mse, self.cfg, j=j, j_max=self.j_max)
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

    async def run(self) -> Path:
        self.prepare()
        state = self.load_state()
        if not state.history:
            # Round-0 certification of the initial incumbent (plan §3.1):
            # without it, a zero-acceptance run would have no stop point.
            self._certify_incumbent(state, "m0")
        while state.round_index <= self.cfg.budget_candidates:
            await self._round(state)
            state = self.load_state()
        _write_json(self.heldout / "done.json", {"rounds": len(state.history), "arm": self.arm})
        return self.rounds_path

    # -------------------------------------------------------------- rounds

    async def _round(self, state: LoopState) -> None:
        assert self.seg_diag is not None and self.train is not None and self.heldout is not None
        self._integrity_check(state.incumbent_id)
        gains = self._feedback_gains()
        feedback_block = self._render_feedback(state, gains)
        system_prompt = (_REPO_ROOT / "experiments" / "prompts" / "system.md").read_text(encoding="utf-8")
        failure_block = ""
        candidate_id = ""
        candidate_seconds: dict[str, float] = {}
        proposal: dict[str, Any] | None = None
        error: str | None = None
        status = "failed"
        diag_mse: float | None = None
        dm_p: float | None = None
        accepted = False
        attempts = 0
        round_seen: set[str] = set()
        tokens_before = workspace_usage(self.ws)
        started = time.monotonic()

        for _attempt in range(1 + MAX_REPAIRS):
            attempts += 1
            round_text = self._render_round(state, feedback_block, failure_block)
            await self._send_with_retries(state, system_prompt, round_text)
            candidate_id, source, proposal, problem = self._collect_candidate(state, round_seen)
            if problem is not None:
                error = problem
                failure_block = problem
                continue
            assert source is not None and proposal is not None
            status, diag_mse, dm_p, accepted, round_error = self._evaluate(
                state, candidate_id, source, proposal
            )
            if status == "evaluated":
                error = None
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
                "proposal": proposal or {},
                "tokens": tokens,
                "feedbackChars": len(feedback_block),
                "attempts": attempts,
                "candidateSeconds": candidate_seconds,
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
    ) -> tuple[str, float | None, float | None, bool, str | None]:
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
                return "failed", None, None, False, f"timed out after {self.cfg.candidate_timeout_seconds:g}s"
            except cand.CheatDetected:
                return "failed", None, None, False, (
                    "batch-invariance check failed (predictions depend on the batch)"
                )
            except (cand.CandidateCrash, cand.ContractViolation) as exc:
                return "failed", None, None, False, str(exc)

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
        seg_a, seg_b = ev.split_ab(seg_cand, self.cfg)
        peaks = select_period_peaks(seg_a.e_full, seg_a.origins, self.cfg)
        channels = seg_cand.values.shape[1]
        posthoc = all_gains(seg_a, seg_b, self.cfg, peaks, level=card_level(self.cfg, peaks, channels))
        _write_json(
            self.heldout / "posthoc" / f"{candidate_id}.json",
            {
                "round": state.round_index,
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

        if not accepted:
            return "evaluated", diag_mse, dm_p, False, None

        # Determinism gate before the incumbent is replaced: the next round's
        # feedback re-fits in-process, and a nondeterministic candidate would
        # make the agent see numbers from a different draw than the ledger.
        module = _load_module("rhg_cand_accept", source)
        adapter = module.create_adapter()
        adapter.fit(self.train, ev.fit_meta(self.cfg, self.values.shape[1], self.seed))
        pred_inproc = ev.predict_adapter(
            adapter, self.values, self.diag_start, self.values.shape[0], self.cfg
        )
        if not np.allclose(pred_inproc, pred_diag, rtol=1e-7, atol=1e-9):
            return "evaluated", diag_mse, dm_p, False, (
                "nondeterministic candidate: in-process re-prediction differs from the "
                "evaluated predictions; seed all randomness from meta['seed']"
            )

        shutil.copyfile(source, self.ws / "models" / "incumbent" / "model.py")
        self._refresh_incumbent_segments()
        self._segments_incumbent_id = candidate_id
        self._certify_incumbent(state, candidate_id)
        return "evaluated", diag_mse, dm_p, True, None


class _SdkRoundRunner:
    def __init__(self, runner: SdkRunner) -> None:
        self._runner = runner

    async def send(self, prompt: str) -> list[Any]:
        return await self._runner.send(prompt)

    async def close(self) -> None:
        await self._runner.close()


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", default="hmm_ar_switch")
    parser.add_argument("--arm", default="card", choices=["scalar", "bj", "card", "placebo"])
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
