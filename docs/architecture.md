# Time-Series Tool-Use Harness Architecture

This project adapts the harness idea from `_reference_project/v3`, but changes the goal.

The system is not a research agent. It does not center on comparing methods, finding a best model, or optimizing baselines. It is a Python-backed tool-use harness with a lightweight static frontend that helps an agent solve time-series tasks by combining background knowledge, data evidence, explicit tool construction, iteration records, and final audited summaries.

## Core Principles

- Bind to DeepSeek Harness as the execution runtime.
- Keep local files and JSONL logs as the source of truth.
- Treat replaceable SDK runtime processes as execution carriers, while DSH persistence and workspace state remain durable.
- Express methodology in Markdown prompts and node specs under `backend/harnessing_ts/config/`.
- Use MCP tools for all structured node state transitions and audit events.
- Make the solution a tool-use process, not a model-selection report.
- Treat training as an implementation detail for tools that require it.

## Implementation Map

The current implementation keeps runtime orchestration, domain routing rules, filesystem state, SDK wiring, and API response shaping in separate modules:

- `backend/harnessing_ts/server.py`: FastAPI app factory, route registration, background task wiring, and static frontend serving.
- `backend/harnessing_ts/api/payloads.py`: aggregate response construction for `/api/bootstrap` and the SSE `live_snapshot` payload.
- `backend/harnessing_ts/api/realtime.py`: SSE fan-out for session parts, builder state, runtime state, and workspace file changes. The browser receives all state updates over the SSE stream.
- `backend/harnessing_ts/orchestrator.py`: application facade that coordinates workspace state, main/node sessions, pending control, knowledge operations, and runner lifecycle.
- `backend/harnessing_ts/node_state.py`: node-chain routing and validation, including `loopDecision`, `nextNode`, pipeline completion, and iterative-solving output requirements.
- `backend/harnessing_ts/agent/sdk_runner.py`: DeepSeek Harness wrapper for JSON-RPC turn execution, session id capture, interrupt/close, and runtime recovery.
- `backend/harnessing_ts/agent/session_factory.py`: SDK runner construction for main and node sessions, including prompt assembly, allowed tools, DSH MCP bridge wiring, and LLM invocation settings.
- `backend/harnessing_ts/mcp/server.py`: stdio MCP bridge and callback binding for governance, audit, runtime settings, knowledge query, and deterministic knowledge-base tools.
- `backend/harnessing_ts/config/deepseek-harness/cordis.yml`: the project composition for the DSH JSON-RPC server, persistence, local filesystem/shell, opt-in Web providers, subagents, tool allowlist, and MCP client.
- `backend/dsh_runtime/`: package-locked Node runtime dependencies and the role/variant-aware DSH tool allowlist plugin. `setup-server` runs `npm ci` and verifies the required runtime entrypoints.
- `backend/harnessing_ts/state/workspace_store.py`: repository facade for workspace JSON/JSONL state, logs, file reads, uploads, resets, runtime settings, and run/artifact records.
- `backend/harnessing_ts/state/workspace_layout.py`: runtime workspace directory layout and DOCX reference text derivatives, extracted in-process by `state/read_docx.py`.
- `backend/harnessing_ts/knowledge_graph.py`: file-backed knowledge-base tables, deterministic tools, graph view/search/cards, and builder/reasoner execution.
- `backend/harnessing_ts/knowledge_prompts.py`: prompt text for the Literature Knowledge Builder and Knowledge Reasoning Agent.
- `backend/harnessing_ts/chain_summary.py`: independent chain builder agent that reads runtime logs, reports, runs, tools, and user artifacts, then writes structured decision-chain JSON for frontend charts and sample evidence cards.
- `frontend/src/main.ts`: static UI rendering and event binding.
- `frontend/src/api.ts`: frontend HTTP helpers and error-message normalization.
- `frontend/src/types.ts`: shared frontend DTO types.

## DSH Runtime, Persistence, And Recovery

`SdkRunner` constructs `DeepSeekHarness` with the repository-local `backend/dsh_runtime` as both `runtime_cwd` and the JSON-RPC runtime entrypoint, and with the workspace `.dsh-sessions/` directory as `session_root`. The Python SDK supplies the client API; `setup-server` must install the project-local Node composition before a live server can start a DSH turn.

Each logical actor has one stable DSH session ID. A runtime process is ephemeral: `session/open` explicitly creates or resumes that ID, executes one prompt, acknowledges `session/checkpoint`, and shuts down. The identity is committed to actor state only after DSH accepts `session/open`; `state/execution.json` retains the reserved identity across a process crash. Recovery uses a resume-first `session/open(mode=recover)`: it creates the same reserved identity only for DSH's exact `session "…" not found` result, and never converts corruption or I/O failures into a new session. The DSH append-only log is the conversation/tool source of truth; the application transcript is only its display projection, and a user message is persisted there only after DSH emits its inbox receipt.

### Pause and restart safety

Normal pause is `agent.cancel({kind:'user'}) → agent.whenIdle() → drainContinuableDescendants() → sessions.flush()`. Only then does the execution state become `paused` and the runtime shut down. The `session/pause` acknowledgement's `checkpointSeq` is copied into `state/execution.json` and the `execution_paused` timeline event, so application state records the same durable boundary as DSH. A per-runner settlement lock serializes normal checkpoint and pause, while expected-runtime close semantics prevent an old turn from closing a newer recovery process. There is no automatic force timeout. A confirmed force-stop is limited to `pausing`; it kills the owned process group, waits for the old pause task to retire, then resumes the same identity and lets DSH persistence repair an incomplete tool boundary.

Every bridged MCP call is fenced by actor, epoch, and lease. `pausing` rejects new calls, while already accepted work remains under DSH cancellation/checkpoint ordering. Async knowledge results revalidate the lease after their await and before audit publication, so a late response cannot mutate a paused or later turn.

`/api/send` accepts prompts only in `ready` or `paused`. Resuming a paused node preserves its node and DSH session identities, starts a new execution epoch, and removes the node's active `pausedAt` marker; historical pause timing remains in `logs/timeline.jsonl`. Pause completion and failure both publish an SSE snapshot; the browser waits for `paused` or `failed` without imposing a force timeout.

## DSH MCP Security Boundary

`dsh-mcp-client` starts `harnessing_ts.mcp.server` over stdio. The bridge forwards stateful calls to the live orchestrator through `/internal/dsh/mcp`, which accepts loopback clients only. Each runner receives a newly generated capability bound to its resolved workspace, role, node session, and allowed `mcp__ts_harness__*` tools; the capability is revoked when the runner closes.

The control URL and capability use the reserved `DSH_MCP_CONTROL_URL` and `DSH_MCP_CAPABILITY` variables. DSH subprocesses scrub `DSH_*` values before model-executed shell commands run. The Cordis MCP configuration explicitly forwards those variables to the bridge process, which is the only intended child recipient. They are not public server settings and must never be surfaced to model prompts, browser payloads, logs, or reverse proxies.

## Node Chain

```text
problem-contract
→ knowledge-to-tools
→ iterative-solving
→ final-summary
```

The main session acts as orchestrator. Each node is a DeepSeek Harness session with its own prompt, tool whitelist, log, and completion summary. The `NOD-KGR-KTL-CRV-SUB-ADA` baseline uses the full four-node chain. Variants can disable specific capabilities; for example profiles without `KTL` skip `knowledge-to-tools`, so their chain becomes `problem-contract → iterative-solving → final-summary`.

## Node Roles

1. `problem-contract`

   Use the user request and references to acquire/process data, explore the data, clarify the actual task, and write both the whole-flow problem contract and the data specification.

2. `knowledge-to-tools`

   Use the problem contract, data specification, references, and the knowledge graph when available to build a deterministic reference feature extractor under `tools/reference-feature-extractor/`. The node must write the evidence map, feature plan, importable extractor, manifest/README contract, real-sample test cases, and evaluation report, then call the backend validator before later nodes can treat the extractor as trusted numeric evidence.

3. `iterative-solving`

   Each round tries exactly one new method, or one explicit combination of previously persisted tools. The method must first be standardized under `tools` with an interface and usage docs, then executed and reviewed through data-first case analysis. The node must write the case review before the iteration summary. Bad-case attribution must start from raw input values or tool features, compare against good cases/prototypes, and only then connect to domain knowledge.

4. `final-summary`

   If iteration is complete, summarize the full optimization trajectory, final tool-use solution, final results, limitations, and evidence.

## Iteration Routing

Node transitions are controlled through MCP, not by parsing markdown artifacts.

- `iterative-solving` still writes `user/iteration-state.md` with `recommend_exit` for auditability.
- To continue, `iterative-solving` must call `mcp__ts_harness__finish_node` with `loopDecision: "continue"` and `nextNode: "iterative-solving"`.
- To exit, `iterative-solving` must call `mcp__ts_harness__finish_node` with `loopDecision: "exit"` and `nextNode: "final-summary"`.
- The backend validates those structured MCP parameters and never uses `user/iteration-state.md` as the control source.

## Filesystem Truth Source

```text
state/workspace.json
state/nodes/<node-session-id>.json
logs/main.jsonl
logs/nodes/<node-session-id>.jsonl
logs/timeline.jsonl
logs/knowledge-graph-builder.jsonl
logs/chain-builder.jsonl
user/problem-contract.md
user/data-spec.md
user/iteration-state.md
user/final-solution.md
state/reference-feature-build.json
artifacts/knowledge-graph.json
artifacts/chain-summary.json
artifacts/
data/processed/
tools/reference-feature-extractor/
tools/
runs/iterations/
reports/iterations/
reports/iterations/<iteration-id>-candidate-review.md
reports/iterations/<iteration-id>-case-review.md
reports/iterations/<iteration-id>-summary.md
reports/final-summary.md
```

The frontend is intentionally thin: it renders the same JSONL logs, node metadata, timeline, and runtime state that the Python backend persists on disk.

## Realtime UI Updates

The backend exposes `/api/events` as a Server-Sent Events stream. While a DSH turn is running, `SdkRunner` turns each root-session notification into one durable, deduplicated transcript part and publishes that part incrementally for main, node, knowledge-graph-builder, and chain-summary-builder sessions. Upload operations publish the refreshed workspace file tree, and task lifecycle events publish runtime snapshots. The stream's bootstrap event provides the complete persisted state for an initial connection or reconnect; the frontend does not poll snapshot endpoints. Messages use keyed DOM reconciliation, and other high-cost panels use content signatures so unchanged UI nodes are retained.

## Chain Summary

The chain summary page is opened from the right rail under the Knowledge Graph entry. Its `Generate` action starts an independent chain builder agent, separate from the main session, node sessions, and knowledge graph builder. The agent reads only runtime workspace logs and artifacts, then writes `artifacts/chain-summary.json` and its trace to `logs/chain-builder.jsonl`.

The JSON output contains metric series over iterations and iteration-level decision-chain cards: proposed methods, test results, sample inspirations, optional visualization paths, next-iteration impact, artifacts, and uncertainty. The frontend renders metric series as charts and sample visualization paths through the workspace file preview endpoint.

## Tool Message Protocol

Every HarnessingTS MCP tool call must include an `intend` argument: one short sentence describing the immediate reason for the call. The bridge requires this field in its JSON schema and strips it before dispatching to business callbacks. DSH native tools follow their own published schemas.

The backend projects assistant messages and `tool/call` / `tool/result` JSON-RPC events into the workspace JSONL transcript. The raw payload of each stored tool event remains available for audit. DSH keeps the complete durable session stream in `.dsh-sessions/`; the HarnessingTS timeline records application-level state and audit effects rather than claiming to duplicate every wire event.

The backend projects DSH `tool/call` / `tool/result` events into intent-bearing, correlated tool-call cards, while retaining the complete JSON-RPC event as the card's raw audit payload. The frontend renders these concise, expandable cards across main, node, knowledge-graph-builder, and chain-summary-builder views; raw protocol events without a projected part are never persisted to the application transcript.

## MCP Boundary

The MCP server exposes structured harness tools while leaving methodology in Markdown prompts and node specs:

- `enter_node`
- `finish_node`
- `record_artifact`
- `record_run`
- `get_runtime_settings`
- `query_knowledge` when the active variant allows it; `KGR` profiles use the built knowledge graph, `RQA` reference-QA profiles answer directly from `references/**`, and `FQA` extractor-source profiles answer from generated `tools/reference-feature-extractor/**` source/rules after static validation.
- reference feature extractor validation/inspection/execution tools when the active variant allows `knowledge-to-tools` and executable extractor use. Some `FQA` profiles still build the extractor but expose it only as static reference material, not as an executable case-review tool.

MCP should not decide which ECG feature, classifier, or LLM judgment strategy is appropriate. It records or mutates harness state, exposes runtime settings, and provides deterministic access to already-built knowledge and reference-feature services.

## ECG5000 Example

For ECG5000 abnormal sample classification, the harness should guide the agent toward this kind of flow:

1. Build `user/problem-contract.md` and `user/data-spec.md` by combining the user request, ECG reference material, data acquisition through `sktime`, processed data schema, exploration, success criteria, evidence rules, and leakage risks.
2. Run `knowledge-to-tools` to turn reference evidence into a validated deterministic feature extractor under `tools/reference-feature-extractor/`, including evidence mapping, feature planning, real-sample tests, control/reference evaluation, and backend validation.
3. Run `iterative-solving` one or more times. Each round builds or revises exactly one tool method, or one explicit combination of existing tools, records results under `runs/iterations/<iteration-id>/`, writes `reports/iterations/<iteration-id>-case-review.md` first, then writes `reports/iterations/<iteration-id>-summary.md`. Case review centers on bad cases: analyze every bad case when there are fewer than 10, otherwise sample 5-20 cases with an explicit strategy. Each reviewed sample needs a visualization path, raw input evidence, current-method evidence, good-case/prototype comparison, and explanation level. Case review then summarizes statistical patterns across all bad cases or the largest computable bad-case set. Summary should focus on method changes, tool paths, aggregate metrics, target gaps, high-level case-review findings, and next-round decisions rather than duplicating per-case details.
4. When the `finish_node` MCP call requests `loopDecision: "exit"` and `nextNode: "final-summary"`, run `final-summary` to summarize the full trajectory and final tool-use solution.
