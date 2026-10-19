# 实验运行 API 契约

前端启动实验、查看运行过程所用的接口。后端实现在 `backend/harnessing_ts/experiments_api.py`，由 `server.py` 注册路由；前端实现在 `frontend/src/experiments.ts`。本文档是两侧共同的接口依据，修改须同步两侧。

## 数据来源

一次运行（run）是 `experiments/runs/<name>/` 目录，由 `experiments/batch.py` 创建：

- `batch-status.jsonl`：每个单元格每次尝试一行，`{"event": "ok"|"fail", "cell", "exitCode", "attempt", "seconds", "at", "detail"?}`。
- `logs/<cell>.log`：单元格进程的输出。
- `<cell>/rounds.jsonl`：每轮一行（字段见 `experiments/loop.py` 的 `_round`）：`round, candidateId, parent, arm, status, diagMse, dmP, accepted, error, evalNote, proposal, tokens, feedbackChars, attempts, candidateSeconds, inspirationParent, judgeCause, judgeReason, judgeFailed, judgeTokens, probesSubmitted, probesKept, probesSeconds, probeOutputsRendered, seconds`。
- `<cell>/heldout.json`：`{"path": "<heldout 目录>"}`；heldout 目录内有 `certification.jsonl`（每次 incumbent 变化一行 `{round, incumbentId, saturated, level, failures[]}`）、`acceptances.jsonl`、`done.json`、`posthoc/<候选>.json`。
- `<cell>/ws/logs/round-<n>.jsonl`：第 n 轮智能体会话的事件日志（公开部件格式见 `agent/translate.py`）。
- 单元格目录名是 `<task>-<arm>-<seed>`；规格来自 `experiments/specs/*.json`。
- 运行目录内的 `batch-process.json`（由 API 写入）：`{"pid": int, "specPath": str, "startedAt": iso, "workers": int, "only": str}`；文件损坏或 pid 不是大于 1 的整数时视为没有进程。
- 运行目录内的 `analysis-process.json`（由 API 写入）：`{"pid": int, "startedAt": iso}`；`analysis-run.log` 是分析进程的输出；`analysis.json` 是 `experiments/posthoc.py` 的输出。
- `<heldout 目录>/posthoc/m0.json`：初始模型的 posthoc 卡片，`diagMse` 是 m0 的诊断 MSE（早期运行可能没有这个文件）。

heldout 数据只给操作者看，页面显示它不会泄露给智能体（智能体只有读写工作区文件的五个工具）。

## 路由

所有路由前缀 `/api/experiments`。运行名称、规格名称与单元格名称必须整串匹配 `^[A-Za-z0-9][A-Za-z0-9._-]*$`（首字符为字母或数字，与 `batch.py` 的名称规则一致）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/specs` | 列出 `experiments/specs/*.json` 中带 `tasks` 字段的规格 |
| GET | `/runs` | 列出 `experiments/runs/` 下的运行目录及摘要 |
| GET | `/runs/{name}` | 一次运行的完整快照（进程状态、单元格矩阵、batch 日志尾部） |
| GET | `/runs/{name}/cells/{cell}` | 一个单元格的轮次表、认证时间线、日志尾部 |
| GET | `/runs/{name}/cells/{cell}/rounds/{n}` | 第 n 轮的提案全文、posthoc 卡片和智能体会话部件 |
| POST | `/runs/{name}/start` | 启动 batch 进程 |
| POST | `/runs/{name}/stop` | 停止 batch 进程 |
| GET | `/runs/{name}/analysis` | 分析进程状态、`analysis.json` 内容、分析日志尾部 |
| POST | `/runs/{name}/analysis` | 在后台子进程里启动 `experiments/posthoc.py` 生成分析 |

## 响应结构

```
SpecSummary      { name, file, description, cellCount, tasks: [str], arms: [str], seeds: [int], budget: int }
RunSummary       { name, process: ProcessStatus, cellsDone, cellsTotal | null, lastActivityAt | null }
ProcessStatus    { state: "running" | "stopped" | "none", pid | null, specFile | null, startedAt | null }

RunSnapshot      { name, process: ProcessStatus, spec: SpecSummary | null,
                   cells: [CellSummary], batchLogTail: str, batchEvents: [BatchEvent] }
CellSummary      { cell, task, arm, seed, roundsDone, budget | null, accepted: int,
                   incumbent: str, saturated: bool | null,
                   state: "pending" | "running" | "done" | "failed",
                   lastDiagMse: number | null, incumbentMse: number | null }
BatchEvent       { event, cell, exitCode, attempt, seconds, at, detail | null }

CellDetail       { cell, task, arm, seed, state, budget | null,
                   rounds: [RoundRow], incumbentCurve: [IncumbentPoint], certifications: [CertRow], logTail: str }
IncumbentPoint   { round: int, mse: number | null, accepted: bool }
RoundRow         { round, candidateId, parent, status, diagMse | null, dmP | null, accepted: bool,
                   error | null, evalNote | null, targetAspect | null,
                   inspirationParent | null, judgeCause | null, judgeFailed: bool,
                   tokens: { inputTokens, outputTokens, cacheReadTokens, reasoningTokens },
                   judgeTokens | null, seconds }
CertRow          { round, incumbentId, saturated: bool, level: number, failures: [{label, ucb, threshold}] }

AnalysisStatus   { process: ProcessStatus,       # specFile 恒为 null
                   analysis: object | null,       # analysis.json 原样解析结果；文件不存在、损坏或不完整时为 null
                   logTail: str }                 # analysis-run.log 尾部

RoundDetail      { round, proposal: object, posthoc: [{label, aspect, delta, lcb, ucb}] | null,
                   parts: [Part] }             # Part = 公开部件，与对话页相同
```

`start` 请求体：`{ "spec": "<规格文件名>", "workers": int (默认 4, 1-16), "only": str (默认 "") }`。成功响应 `{ "process": ProcessStatus }`。同一个运行名称已有进程时返回 409。运行名称对应的目录不存在时创建。`only` 直接传给 `experiments/batch.py`，其中 `cell=<值>` 按单元格名子串匹配；需要精确匹配时用完整 cell id。

`start` 的补充规则：运行目录里已有 `batch-process.json` 且其 `specPath` 的文件名与请求的 `spec` 不同，则返回 409（一个运行目录对应一份实验协议，换规格请换运行名称），与进程是否存活无关。

`stop` 向进程组发送 SIGTERM（batch 收到后会终止所有单元格子进程，宽限 5 秒），并等待进程退出最多 8 秒；响应 `{ "process": ProcessStatus }`，其中 `process.state` 仍可能是 `running`（超时未退出）。没有进程时返回 409。

`GET /runs/{name}/analysis` 返回 `AnalysisStatus`。`process.state` 为 `running` / `stopped` / `none`（从未启动过分析）。`analysis` 是 `analysis.json` 原样的解析结果（`endpointTable`、`contrasts`、`runs`、`incomplete`、`audit` 等，字段以 posthoc 输出为准，API 不改写）；半写入或损坏的文件得到 `null`，不会报 500。

`POST /runs/{name}/analysis` 无请求体，成功响应 `{ "process": ProcessStatus }`。API 在后台子进程（独立会话）里运行 `uv run --extra research python experiments/posthoc.py --runs <运行目录> --out <运行目录>/analysis.json`，输出写入 `<运行目录>/analysis-run.log`（每次启动覆盖），pid 记入 `analysis-process.json`，请求线程内不导入也不调用 posthoc。同一次运行同时只允许一个分析进程，已有存活进程返回 409。真实任务需要 CSV：如果 `batch-process.json` 记录了 `specPath`，则对该规格里每个 `kind` 为 `real` 的任务追加 `--real-csv <任务名>=<仓库根目录>/<该任务的 csv>`；没有记录规格时不追加。分析失败时 `process.state` 为 `stopped` 且 `analysis` 为 `null`，原因在 `logTail` 里；每次启动分析前先删除旧的 `analysis.json`，所以旧结果不会被当作新结果。

增量模型曲线与摘要：`incumbentCurve` 每个轮次下标一个点，从第 0 轮到 `rounds.jsonl` 里最后一轮。第 0 轮是初始模型 m0，`mse` 为 `<heldout>/posthoc/m0.json` 的 `diagMse`（文件不存在时为 `null`），`accepted` 恒为 `false`。之后每一轮的 `mse` 是该轮结束后当前 incumbent 的诊断 MSE：该轮被接受则是候选自己的 `diagMse`，否则沿用上一点的值；`accepted` 为 `true` 表示该轮 incumbent 发生了变化。`rounds` 每行的 `diagMse` 是该候选自己的诊断 MSE。`CellSummary.incumbentMse` 等于曲线最后一点的 `mse`（最后一次被接受候选的 `diagMse`，没有则 m0 的 `diagMse`，再没有则 `null`）；`lastDiagMse` 是最后一个有数值的候选 `diagMse`。

状态判定：进程是否存活由 `batch-process.json` 中的 pid 检查：本服务启动的子进程看其退出状态；服务重启后，pid 必须存在、有权限访问且是自己进程组的组长（启动时使用新会话，所以 pid 等于 pgid；这排除了大部分 pid 被复用的情况，但排除不了复用者恰好也是进程组组长的情形）。单元格状态按下列顺序判定：

1. `done`：heldout 目录下有 `done.json`（与 `batch.py` 的 `cell_done` 一致）。
2. `running`：batch 进程存活，并且 `logs/<cell>.log` 的修改时间不早于进程的 `startedAt`，也晚于该单元格最后一条 `batch-status.jsonl` 事件的时间（事件时间只精确到秒，所以比较的是日志修改时间取整到秒）。
3. `failed`：该单元格最后一条 `batch-status.jsonl` 事件为 `fail`，且不是 `running`。
4. `pending`：其余情况（还没轮到的单元格、被 `--only` 排除的单元格、被停止后留下 `rounds.jsonl` 的单元格）。

## 错误

非法名称 400；不存在的运行/单元格/轮次 404；状态冲突 409。错误体沿用服务现有的 `{"detail": str}`。
