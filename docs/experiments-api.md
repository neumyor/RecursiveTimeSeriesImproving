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
- 运行目录内的 `batch-process.json`（由 API 写入）：`{"pid": int, "specPath": str, "startedAt": iso, "workers": int, "only": str}`。

heldout 数据只给操作者看，页面显示它不会泄露给智能体（智能体只有读写工作区文件的五个工具）。

## 路由

所有路由前缀 `/api/experiments`。运行名称与规格名称只允许字母、数字、`.`、`_`、`-`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/specs` | 列出 `experiments/specs/*.json` 中带 `tasks` 字段的规格 |
| GET | `/runs` | 列出 `experiments/runs/` 下的运行目录及摘要 |
| GET | `/runs/{name}` | 一次运行的完整快照（进程状态、单元格矩阵、batch 日志尾部） |
| GET | `/runs/{name}/cells/{cell}` | 一个单元格的轮次表、认证时间线、日志尾部 |
| GET | `/runs/{name}/cells/{cell}/rounds/{n}` | 第 n 轮的提案全文、posthoc 卡片和智能体会话部件 |
| POST | `/runs/{name}/start` | 启动 batch 进程 |
| POST | `/runs/{name}/stop` | 停止 batch 进程 |

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
                   lastDiagMse: number | null, bestDiagMse: number | null }
BatchEvent       { event, cell, exitCode, attempt, seconds, at, detail | null }

CellDetail       { cell, task, arm, seed, state, budget | null,
                   rounds: [RoundRow], certifications: [CertRow], logTail: str }
RoundRow         { round, candidateId, parent, status, diagMse | null, dmP | null, accepted: bool,
                   error | null, evalNote | null, targetAspect | null,
                   inspirationParent | null, judgeCause | null, judgeFailed: bool,
                   tokens: { inputTokens, outputTokens, cacheReadTokens, reasoningTokens },
                   judgeTokens | null, seconds }
CertRow          { round, incumbentId, saturated: bool, level: number, failures: [{label, ucb, threshold}] }

RoundDetail      { round, proposal: object, posthoc: [{label, aspect, delta, lcb, ucb}] | null,
                   parts: [Part] }             # Part = 公开部件，与对话页相同
```

`start` 请求体：`{ "spec": "<规格文件名>", "workers": int (默认 4, 1-16), "only": str (默认 "") }`。成功响应 `{ "process": ProcessStatus }`。同一个运行名称已有进程时返回 409。运行名称对应的目录不存在时创建。

`stop` 向进程组发送 SIGTERM；响应 `{ "process": ProcessStatus }`。没有进程时返回 409。

状态判定：进程是否存活由 `batch-process.json` 中的 pid 检查（`os.kill(pid, 0)`）；单元格 `done` 由 heldout 下 `done.json` 判定，`failed` 由 `batch-status.jsonl` 最后一条该单元格事件为 `fail` 且进程未再启动它判定，有 `rounds.jsonl` 但未完成且进程在运行判为 `running`，否则 `pending`。

## 错误

非法名称 400；不存在的运行/单元格/轮次 404；状态冲突 409。错误体沿用服务现有的 `{"detail": str}`。
