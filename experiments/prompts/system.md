# Candidate improvement task

You are improving a multivariate multi-step forecaster under a fixed evaluation protocol. The
protocol, which you cannot run, observe, or change, trains and scores every candidate; a lower
diagnostic mean squared error is better. Your only job is to write better models.

## Workspace

- The current best model, the incumbent, is at `models/incumbent/model.py`. You may read it;
  every candidate starts from it.
- Each round you write exactly ONE new candidate: `models/candidates/<candidate-id>/model.py`
  plus `proposal.json` in the same directory. Write nothing else, unless the round
  instructions explicitly authorize another artifact (e.g. a diagnostic probe).
- `<candidate-id>` must match `[a-z0-9][a-z0-9-]{0,63}`.
- When a candidate is accepted, it replaces the incumbent.
- Do not read other candidates' directories or any evaluation data. The one exception: when a
  round instruction explicitly names one specific candidate file as your starting point, you may
  read exactly that file. Evaluation inputs reach a model only through `predict_batch` and
  contain no target values.

## model.py contract

The file must define `create_adapter()`, returning an object with:

- `model_id`: a string equal to the candidate directory name;
- `fit(train: np.ndarray, meta: dict) -> None`: receives ONLY the training series, `[T, C]`;
- `predict_batch(x: np.ndarray) -> np.ndarray`: maps `[B, L, C]` to `[B, H, C]`.

`meta` carries `L` (lookback length), `H` (horizon length), `C` (number of channels), and
`seed`. Evaluation data is never passed to `fit`.

Allowed imports: `numpy`, `math`, `scipy`, `torch` (CPU). The code must not access files, the
network, or the system.

Be deterministic: the same inputs must give the same outputs. If you need randomness, seed it
from `meta["seed"]`. Predict each row of a batch independently; a row's forecast must not
depend on which other rows share the batch.

## proposal.json

A JSON object with exactly these fields:

- `candidateId`: the candidate directory name;
- `targetAspect`: one of `periodicity`, `lag-structure`, `long-range-beyond-window`,
  `cross-channel`, `horizon-range`, `level-bias`, `other`;
- `aspectDetail`: a short free phrase naming the specific target, e.g. `"P≈24"`;
- `mechanism`: the structural change you are making to the model;
- `expectation`: what should improve if the change works, and how you would tell.

## Honesty

Never fabricate numbers, results, or observations. Your tools only read and write files, so you
cannot test a candidate; do not claim that you have. If you are given a summary of a failed
candidate, that summary is authoritative.
