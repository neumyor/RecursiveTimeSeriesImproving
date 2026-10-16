# Building your own diagnostic probes

Besides your candidate, you may write diagnostic probes: small analysis tools
that the harness runs for you on the incumbent's data. A probe is pure
analysis — it cannot touch the incumbent, the training data, or scoring. It
returns a small JSON summary that is shown back to you every round. This is
your only diagnostic channel beyond the scalar numbers: build it well and it
tells you where the incumbent still fails; build nothing and you are working
blind.

## Files

- `probes/<probe-id>/probe.py` and `probes/<probe-id>/probe_proposal.json`.
- `<probe-id>` must match `[a-z0-9][a-z0-9-]{0,63}`.
- Probe ids are never reused or edited in place: to revise a probe, write a
  new one under a new id. Old probes keep running until superseded.

## probe.py contract

The file must define `create_probe()`, returning an object with:

- `probe_id: str` — must equal the probe directory name;
- `run(artifacts: dict) -> dict`: returns your diagnostic summary — a flat,
  JSON-serializable dict of at most a few KB. Put the numbers you want to
  watch every round in it.

`artifacts` contains, computed from the CURRENT incumbent on the diagnostic
segment:

- `residuals["e"]`: `[T, C]` float array — the incumbent's realized 1-step
  forecast-error series (its own past mistakes; NaN-padded at head and tail).
- `residuals["r"]`: `[n, H, C]` float array — the incumbent's H-step residuals
  per forecast origin, aligned with `residuals["origins"]`.
- `activations`: dict mapping layer name to a `[m, P, D]` float32 array of the
  incumbent's hidden-layer activations on `m` evenly spaced diagnostic
  windows. `activations[name][i]` belongs to the same forecast as residual
  row `meta["activationOrigins"][i]` of `residuals["r"]` — so you can ask
  which internal states precede which errors. Empty dict if the incumbent has
  no activation path.
- `meta`: `{"L", "H", "C", "T", "seed", "incumbent", "task", "activationOrigins"}`.

Allowed imports: `numpy`, `math`, `scipy`. No file, network, or system access.
`run` must be deterministic: identical calls return identical dicts.

## probe_proposal.json

A JSON object with exactly these fields, all non-empty strings:

- `probeId`: the probe directory name;
- `kind`: `"residual"` if the probe reads `residuals`, `"activation"` if it
  reads `activations`;
- `measures`: a short human phrase naming what it measures, e.g.
  `"dominant residual period"`;
- `signalKey`: the key of the primary number in your output dict — the one
  your probe is judged on;
- `expectedDirection`: `"higher"` or `"lower"` — when the structure your probe
  looks for is genuinely present, this number must move in that direction;
- `rationale`: why this measurement should guide a model change.

## Validation, then trust

New probes never touch your data first. They are validated on synthetic
fixtures with KNOWN planted structure — residual fixtures plant periodic
components and cross-channel lag coupling at undisclosed parameters;
activation fixtures plant structure at a known layer of a fixed small
network. Your probe is kept only if its `signalKey` number moves in the
declared direction on the planted fixtures and the probe is deterministic.
The verdict appears in your next round's feedback. A validated probe is
re-run on every future incumbent and its output is shown to you every round.

A probe that times out (120 s), crashes, exceeds the output cap, or returns
nondeterministic results is rejected with the reason in your feedback. Fix it
under a new id.
