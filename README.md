# RecursiveTimeSeriesImproving

RecursiveTimeSeriesImproving is a research harness for residual-headroom experiments on time-series forecasting. The web console starts and monitors experiment runs (`experiments/loop.py`), and the Settings page edits the main model configuration used by the agents.

It is not a leaderboard or model-comparison framework. The goal is a structured, auditable search for forecasting improvements.

## Quick Start

### 1. Install Prerequisites

You need Python 3.11+, Git, [uv](https://docs.astral.sh/uv/), Node.js 20+ with npm, [Bun](https://bun.sh/) for the frontend, and a DeepSeek-compatible API credential. `uv sync` installs the official DeepSeek Harness Python SDK. Runtime execution uses the project-local, package-locked Node composition in `backend/dsh_runtime/`, which contains the JSON-RPC server, Cordis composition, MCP client, filesystem/shell tools, Web plugins, and subagent plugins.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

### 2. Prepare The Project

```bash
git clone https://github.com/neumyor/RecursiveTimeSeriesImproving.git
cd RecursiveTimeSeriesImproving

uv sync
uv run ts-harness setup-server
```

`setup-server` installs and verifies the project-local DeepSeek Harness Node runtime, builds the production frontend, and prepares a shared runtime base for task workspaces. The Python SDK is configured to launch this local runtime explicitly, rather than relying on a bundled runtime. To refresh it independently, run `npm ci --prefix backend/dsh_runtime`. Do not update the Python SDK or `backend/dsh_runtime` packages independently; run the full test suite after a coordinated runtime upgrade.

### 3. Configure The LLM

Choose one authentication mode.

Use DeepSeek Harness environment credentials:

```bash
export TS_HARNESS_LLM_AUTH_MODE=sdk-default
export DEEPSEEK_API_KEY=YOUR_API_KEY
```

Or save an explicit DeepSeek-compatible endpoint configuration:

```bash
export TS_HARNESS_LLM_AUTH_MODE=manual
export TS_HARNESS_LLM_PROTOCOL=deepseek
export TS_HARNESS_LLM_MODEL=deepseek-v4-flash
export TS_HARNESS_LLM_BASE_URL=https://api.deepseek.com
export TS_HARNESS_LLM_API_KEY=YOUR_API_KEY
```

Check the effective configuration without printing the full API key:

```bash
uv run ts-harness llm-config
```

### Optional Web Tools

Filesystem, shell, and `SUB` subagent tools are supplied by the local DSH runtime and filtered by the active node/variant. Web tools are disabled unless the deployment opts in:

```bash
# Only enable where the server is not able to reach sensitive internal hosts.
export TS_HARNESS_ENABLE_WEB_FETCH=true

# Reuses DEEPSEEK_API_KEY for search.
export TS_HARNESS_ENABLE_WEB_SEARCH=true
```

The current upstream HTTP fetch provider does not provide private-network/SSRF protection, so do not enable `TS_HARNESS_ENABLE_WEB_FETCH` on a host that can access trusted internal services.

### 4. Start Locally

Use a separate workspace per experiment or ablation variant.

```bash
export TS_HARNESS_WORKSPACE="$HOME/.harnessingts/workspaces/default"
export TS_HARNESS_VARIANT=NOD-KGR-KTL-CRV-SUB-ADA
export TS_HARNESS_CONTROL_MODE=auto
export HOST=127.0.0.1
export PORT=4327

uv run ts-harness-server
```

Open [http://127.0.0.1:4327](http://127.0.0.1:4327).

Useful launch options:

```bash
# Show registered ablation profiles.
uv run ts-harness variants

# Run the UI without calling DeepSeek Harness.
TS_HARNESS_DRY_RUN=true uv run ts-harness-server

# Enable debug/reset controls. Do not use this on shared/public servers.
TS_HARNESS_DEBUG=true uv run ts-harness-server
```

## Using The UI

The web console has two pages:

1. **实验运行** (`/experiments`): choose a registered spec under `experiments/specs/`, pick all cells or one feedback arm, set the number of parallel workers, and start the run. The matrix shows each cell's rounds and accepted candidates; a cell page shows its diagnostic MSE curve, round table and saturation certificates. **生成分析** runs the held-out posthoc analysis for finished cells.
2. **设置** (`/settings`): read the workspace state and edit the main model configuration.

One cell is one task × one feedback arm × one seed, and runs for the spec's round budget. Runs are resumable: restarting the same run name skips cells whose `done.json` exists.

### Session Recovery

DSH's append-only session log under `.dsh-sessions/` is the conversation and tool-history source of truth. Each logical main/node actor owns one stable DSH session ID. Every turn opens a fresh runtime process, explicitly creates or resumes that identity, runs one prompt, flushes a durable checkpoint, and closes the process.

Pause uses DSH's native boundary: cancel the root agent, wait until it is idle, drain continuable descendants, flush the session, then acknowledge `paused`. The acknowledged DSH checkpoint sequence is persisted in `state/execution.json` and in the `execution_paused` timeline event. It has no automatic client or server timeout. A separately confirmed force-stop is available only while `pausing`; recovery first resumes the same session so DSH can repair an incomplete tool boundary. If a process died after reserving an identity but before DSH created its log, recovery may create that exact identity only after DSH explicitly reports it absent. Continuing a paused actor requires a non-empty user message; it clears the node's active `pausedAt` marker while retaining the historical pause event in the timeline. `running`, `pausing`, `recovering`, and `failed` executions reject new messages.

## Ablation Variants

`TS_HARNESS_VARIANT` is read once at process startup. Restart the server to switch variants, and use a different `TS_HARNESS_WORKSPACE` for each experiment.

Feature codes are order-insensitive. For example, `RQA-NOD-SUB-ADA` is accepted and canonicalized to `NOD-RQA-SUB-ADA`.

```bash
uv run ts-harness variants
```

Feature codes:

| Code | Meaning |
|---|---|
| `NOD` | RecursiveECG node chain. |
| `KGR` | File-backed reference knowledge graph and graph-backed `query_knowledge`. |
| `RQA` | Direct reference QA agent over `references/**`; mutually exclusive with `KGR`. |
| `FQA` | QA over generated `tools/reference-feature-extractor/**` source/rules as replacement reference material; mutually exclusive with `KGR` and `RQA`. |
| `KTL` | `knowledge-to-tools` node and validated reference feature extractor. |
| `CRV` | Structured case review and case-level attribution. |
| `CXR` | Structured case review with reference feature extractor execution disabled. |
| `SUB` | Independent candidate testing/review through DSH subagents. |
| `ADA` | Adaptive iterative solving and stop decisions. |
| `DIR` | Direct single-agent baseline; cannot be combined with other codes. |

Registered profiles:

| Canonical ID | Description |
|---|---|
| `NOD-KGR-KTL-CRV-SUB-ADA` | Full RecursiveECG. |
| `NOD-RQA-KTL-CRV-SUB-ADA` | Full node workflow without the knowledge graph; `query_knowledge` answers directly from references. |
| `NOD-RQA-KTL-SUB-ADA` | Direct reference QA with case review removed. |
| `NOD-RQA-CRV-SUB-ADA` | Direct reference QA with `knowledge-to-tools` and reference feature extractor removed. |
| `NOD-RQA-SUB-ADA` | Direct reference QA with both case review and knowledge tools removed. |
| `DIR` | Ordinary single-agent tool-use baseline with no node chain or harness MCP tools. |
| `NOD-FQA-KTL-CXR-SUB-ADA` | Builds the reference feature extractor, then uses its source/rules as downstream QA material; case review remains but extractor execution is disabled. |

A removed feature is not merely optional. It is blocked by tool availability and backend checks where applicable. For example, no-`KTL` profiles cannot enter `knowledge-to-tools`, cannot validate or run the reference feature extractor, and do not expose extractor metadata through the API.

## Startup Errors

The server validates startup-only configuration before binding the port.

| Error text | What to do |
|---|---|
| `Invalid PORT` | Set an integer from 1 to 65535, for example `PORT=4327`. |
| `Invalid TS_HARNESS_CONTROL_MODE` | Use `auto` or `manual`. |
| `unknown feature code` | Run `uv run ts-harness variants` and choose a listed code. |
| `DIR cannot be combined` | Use `DIR` alone. |
| `KGR, RQA, and FQA are mutually exclusive` | Choose graph-backed knowledge, direct reference QA, or extractor-source QA. |
| `syntactically valid but not a registered ablation profile` | Use one of the registered canonical profiles. |

## Server Deployment

For LAN or server use, bind to a non-loopback address and keep all startup-only values in the same service environment.

```bash
export HOST=0.0.0.0
export PORT=4327
export TS_HARNESS_WORKSPACE=/srv/RecursiveECG/workspaces/nod-kgr-ktl-crv-sub-ada
export TS_HARNESS_VARIANT=NOD-KGR-KTL-CRV-SUB-ADA
export TS_HARNESS_CONTROL_MODE=auto

uv run ts-harness-server
```

One-line launch example:

```bash
HOST=0.0.0.0 \
PORT=4327 \
TS_HARNESS_WORKSPACE=/srv/RecursiveECG/workspaces/nod-rqa-sub-ada \
TS_HARNESS_VARIANT=NOD-RQA-SUB-ADA \
TS_HARNESS_CONTROL_MODE=auto \
uv run ts-harness-server
```

The server prints a LAN URL when it can detect one. Open the port in your firewall if needed.

For long-running deployments, use a process manager such as `systemd`, `supervisord`, Docker, or tmux. Example `systemd` fragment:

```ini
[Service]
User=harness
WorkingDirectory=/path/to/RecursiveECG
Environment=BUN_INSTALL=/home/harness/.bun
Environment=PATH=/home/harness/.bun/bin:/usr/local/bin:/usr/bin:/bin
Environment=HOST=0.0.0.0
Environment=PORT=4327
Environment=TS_HARNESS_WORKSPACE=/srv/RecursiveECG/workspaces/nod-kgr-ktl-crv-sub-ada
Environment=TS_HARNESS_VARIANT=NOD-KGR-KTL-CRV-SUB-ADA
EnvironmentFile=/etc/RecursiveECG.env
ExecStart=/usr/bin/env uv run ts-harness-server
Restart=on-failure
```

Keep API keys in an environment file with restricted permissions. Put public deployments behind authentication, HTTPS, and network allowlisting.

### Internal MCP Bridge

The DSH MCP bridge is an implementation detail, not a public integration endpoint. For each runner, the server creates a short-lived capability bound to the workspace, session role, node ID, and allowed MCP tools. The bridge calls the loopback-only `/internal/dsh/mcp` endpoint; it is rejected for non-loopback clients.

`DSH_MCP_CONTROL_URL` and `DSH_MCP_CAPABILITY` are reserved internal variables. Cordis forwards them only to the stdio MCP child process, while DSH removes `DSH_*` values from model-executed shell children. Do not expose these values, set them in prompts, or route the internal endpoint through a reverse proxy.

## Workspaces And Runtime Files

Runtime workspaces are separate from the source repository. By default they live under:

```text
~/.harnessingts/workspaces/default
```

Important workspace directories:

```text
references/                 uploaded reference files
data/raw/                   uploaded raw task data
data/processed/             derived task data
user/                       problem contracts, data specs, final solution
tools/                      generated tools and reference feature extractor
runs/                       iteration run outputs
reports/                    candidate/case/final reports
knowledge_base/             knowledge graph tables and cache
state/                      JSON state
logs/                       JSONL logs
.dsh-sessions/              durable DSH JSON-RPC session events/checkpoints
```

SDK sessions run with `cwd` set to the runtime workspace. Task dependencies should be installed into that workspace, not into the RecursiveECG source repository:

```bash
uv run python script.py
uv add package-name
uv run --with package-name python script.py
```

## Development Checks

For local code changes:

```bash
uv run python -m compileall -q backend/harnessing_ts
uv run pytest -q

cd frontend
bun install --frozen-lockfile
bun run build
```

The Vite development UI is available at `http://127.0.0.1:5173`:

```bash
cd frontend
bun run dev
```
