# Residual Headroom 实验系统：最终编码计划

> 金文档：[minipaper v0.2](minipaper-residual-guided-discovery-v0.2.zh-CN.md)
> 本计划综合了初版方案 [residual-headroom-implementation-plan](residual-headroom-implementation-plan.zh-CN.md) 与三方评审（A 研究有效性 / B 代码集成 / C 范围与排期）的意见，以下内容以本文为准。
> 日期：2026-10-15

---

## 0. 相对初版的三项关键决策

| # | 决策 | 依据 |
|---|---|---|
| D1 | **论文实验用独立循环 `experiments/loop.py`，直接调用已有的 `SdkRunner`（DSH SDK + 本地 runtime + 会话持久化 + transcript），不经过 orchestrator、MCP、节点链和变体系统**。harness 集成（RHG 变体等）推迟到投稿后（P5） | C：实验控制更强、可以并行、论文伪代码就是代码。B：harness 集成有 8 个严重问题（分发入口、300s MCP 超时、guidance 叠加、`run-rhg` 不可行、环境变量泄漏、`recommend_exit` 双重约束等），每一个都是进度风险 |
| D2 | **从 `dsh-backend-develop` 开 `research/residual-headroom`，不从 codex 分支起步**。codex 里需要的代码（线性 ridge、killpg 子进程、`create_adapter` 加载）合计不到 200 行，复制改写即可 | B：codex 分支根目录提交了约 113MB 的 PFO 产物；它的 adapter、runner 都会连带 import profiling/torch 链路，`_run_profile_process` 也不能原样复用。C：不必押注 38 个未推送的提交 |
| D3 | **评估、接受、认证全部由循环在 agent 之外执行。agent 每轮只写一个候选，不能调用评估，也没有 bash** | C：agent 拿到 bash 就能自己算诊断，冲淡各臂差异，也会放大泄漏面。A/B：预算和停止应由后端决定 |

**复用了什么**：`agent/sdk_runner.py::SdkRunner`、`settings/llm.py`（LLM 配置）、`state/message_log.py`（transcript）、`backend/dsh_runtime` + `cordis.yml` + `tool-allowlist.mjs`（工具白名单）、`.dsh-sessions` 中的 usage 记录（token 计量），以及从 codex 分支改写的三段代码。KTL 的“生成 → 校验”机制留到 P5，用于领域坐标。

## 0.5 执行约束（2026-10-15 与作者确认，凌驾于其余章节）

1. **目标是一篇 ICML 2027 论文，不是一个产品系统。** 一切决策以“尽快得到完整系统与论文所需实验结果”为准；KDD 作为备选。
2. **取消先导实验（原 P2/G2 go/no-go）。** G0 证书校准保留（理论正确性检验，成本低，必须通过）。系统完成后直接进入主实验。
3. **多代理并行开发**，同时最多 8 个子代理；每个大里程碑（M1 核心层、M2 循环系统、M3 实验与结果）完成后启动三方代码评审，检查是否符合本计划与约束。
4. **工程原则：正确性 > 简洁性 > 可扩展性。** 禁止 over-engineering：不做文件树 hash 校验、源码大小限制等不影响实验正确性的硬检查；AST 白名单保留最小版本。**保留异常分类与恢复机制**（失败候选计入预算、修复回合、子进程超时回收、按轮断点续跑），这是系统健康运行的关键保障。
5. **全程自主推进直到系统完整可运行**，无需作者逐项确认；仅在方向性风险（G0 不达标、实验否定核心假设）出现时汇报。

接口层面的权威规格见 [headroom-contracts.md](headroom-contracts.md)。

---

## 1. 目录结构

```
backend/harnessing_ts/headroom/          # 核心层：纯 numpy/scipy，确定性，不依赖 LLM 或 harness
├── __init__.py
├── config.py        # 冻结配置（camelCase JSON ↔ frozen dataclass，validate 汇总报错）
├── timeidx.py       # 时间索引约定、切分、embargo、起点枚举
├── synthetic.py     # 合成过程 + 全信息 oracle
├── residuals.py     # 残差张量；已实现信息集的特征构造
├── correctors.py    # 坐标枚举；线性校正器；held-out 实现增益
├── inference.py     # HAC/DM(HLN)、解析置信界、块 bootstrap（仅展示用）
├── certify.py       # 多坐标 Bonferroni + 均匀 α 的 ε-饱和
├── feedback.py      # Scalar / BJ / Card / Placebo 四种渲染
├── candidate.py     # 候选契约、AST 白名单、隔离子进程评估、批不变性检查
└── usage.py         # 读取 .dsh-sessions/**/session.jsonl.zstd 中的 usage

experiments/
├── loop.py          # 独立 agent 循环（每轮一个新的 DSH 会话）
├── prompts/         # system.md、round.md（四臂共用，可以全文放进论文附录）
├── templates/m0/model.py   # 初始父模型模板（从 codex linear_forecast 改写，约 40 行）
├── stage0.py        # 证书校准与任务筛选（不涉及 LLM）
├── baselines.py     # 一次性 m0+g、规则修复、auto.arima/ETS
├── batch.py         # 任务 × 臂 × run 的并行调度与断点续跑
├── posthoc.py       # 离线指标、审计、统一 m+g 后处理
├── figures.py
└── specs/*.json     # 冻结的实验规格

tests/test_headroom_*.py
```

**依赖**：新增 optional 组 `research = [numpy>=2, scipy, torch, zstandard, statsforecast]`，不影响主服务的安装。

---

## 2. 核心层规格

### 2.1 时间索引约定（`timeidx.py`）

- **起点 o 是窗口中最后一个观测的下标**，输入为 $x_{o-L+1..o}$，目标为 $x_{o+1..o+H}$，残差 $r[o,h]=x_{o+h}-\hat x_{o+h|o}$，$h=1..H$。
  - 注意：codex `iter_target_windows` 中的 `split = start+L` 是第一个目标的下标，**不能直接当作 o 使用**。
- 主实验取 **H=24**，L=96（C、A-M8：样本量更大，计算量更小，也更接近 h=1 的理论）。H=96 只放进补充实验。
- **切分**：
  - 合成任务：train 和 diag 来自同一条主序列；**cert 和 test 来自同一过程的独立新模拟段**，长度足够（cert ≥ 2 万步，test ≥ 5 万步），并带 burn-in。
  - 真实任务：按时间顺序切成 train / diag / cert / test，各段之间加 **H 步 embargo**。
- diag 内部按时间切成 A 和 B，中间同样加 H 的 embargo。
- **burn-in**：残差序列只从 diag 起点开始计算；前 $D$ 个起点不用于拟合校正器，其中 $D=\max(\text{lag 上限},\ \text{FIR 长度})$。
- 标准化统计量只用 train 计算。

### 2.2 合成任务与 oracle（`synthetic.py`）

| 任务 | P0 | oracle |
|---|---|---|
| `lgssm_multiperiod`：周期 24 和 168，部分观测 | ✓ | 从序列起点运行 Kalman 滤波，再做 h 步外推（全信息） |
| `lgssm_xchannel`：跨通道变滞后耦合，C≤8 | ✓ | 同上 |
| `hmm_ar_switch`：Hamilton 式 AR 体制切换，观测无噪声 | ✓ | 联合递推 $E[x_{o+h}\mathbf 1\{S_{o+h}=s\}\mid\mathcal I]$，复杂度 $O(K^2h)$，**不能用“按后验混合各体制预测”**（A-M9） |
| `arfima` | ✓（检验 HAC 能否扛住长记忆） | DL/innovations 算法，阶数 ≥ D（例如 2000） |
| `garch`、`random_walk`（负对照） | ✓ | 0；最后一个值 |
| **每个任务都做 m = oracle 的负对照** | ✓ | — |
| `bilinear`（$x_t=b\,x_{t-2}e_{t-1}+e_t$）、`tar` | P4 | 每个起点 ≥1000 条路径的蒙特卡洛；单测断言样本 ACF≈0 |

- **oracle 的信息深度 ≥ D**（A-S6）。“L-受限 oracle”和“正确设定模型在同一训练集上的拟合误差”只作为次要参照。
- **gap 的计算**：oracle 预测和候选预测用同一套标准化，在**同一批起点**上配对计算经验 MSE，$\text{gap}=\mathrm{MSE}(m)-\mathrm{MSE}(\text{oracle})$。

### 2.3 已实现信息集与坐标（`residuals.py`、`correctors.py`）

所有特征都必须是 $\mathcal I_o$ 可测的，由**截断不变性测试**保证：把数组截断到 o 后算出的特征，必须与用全长数组算出的完全一致。

| 坐标 | 校正器输入（滞后均**相对起点 o**，k 从 0 开始） |
|---|---|
| `lag[k1,k2]` | $e_{o-k}$，其中 $e_t=r[t-1,1]$ 为最新已实现的 1 步残差。默认分段 `[0,23]`、`[24,95]`、`[96,335]` |
| `horizon` | 同步长的已实现残差 $r[o-h-k,h]$，$k=0..3$（A-M3），对每个 h 单独拟合 |
| `band(P)` | $e$ 经**因果** FIR 带通后取两个正交滞后（0 和 P/4，保留相位信息）。周期峰**只用 diag-A 的残差选取**（A-M4） |
| `beyond-L` | **只用原始值** $x_{o-k}-x_o$（$k\ge L$），避免和 lag 坐标混淆（A-M11） |
| `channel(→i)` | 其他所有通道的 $e$ 与 $\Delta x$ 的过去，用来预测通道 i 的残差，共 C 个坐标（C：不做 log-det 筛选） |
| `total` | 以上全部特征的并集 |

- **校正器**：ridge 回归，α 只在 A 内做带 H 间隔的时间序列 CV 来选；在 A 上拟合，在 B 上评估实现增益。
- **实现增益**：逐起点损失差 $d_o=\frac1{HC}\sum_{h,c}[r^2-(r-\hat g)^2]$，$\hat\Delta=\bar d$。同时报告 `by_horizon` 和均值偏移项 μ²。
- **置信界**：Newey–West HAC 方差（带宽自动选择且 ≥H−1）+ HLN 小样本修正 + t 临界值，得到解析的 LCB/UCB。块 bootstrap 只用于卡片中的展示区间。
- **卡片排序**用 Bonferroni 调整后的 LCB，抑制头号坐标的赢家诅咒（A-M4）。
- 不做：多变量 Szegő 筛选、IS 密度、multitaper KS 闭式（只在 stage0 的 notebook 中验证观察 0 和命题 1 的闭式）。

### 2.4 认证（`certify.py`）

- **何时检验**：只在 incumbent 改变时检验，即第 0 轮和每次接受之后（A-S4）。被拒的轮次不花 α。
- **α 分配**：$\alpha_j=\alpha/J_{\max}$，其中 $J_{\max}=\text{预算}+1$。
- **统计量集合** $\mathcal K$ = {total} ∪ 各坐标 ∪ 两个步长组；每一项的置信水平为 $1-\alpha_j/|\mathcal K|$（论文 §3.5 的 Bonferroni，A-S3）。
- **判停**：所有 $k\in\mathcal K$ 都满足 $\mathrm{UCB}_k<\varepsilon\cdot\mathrm{MSE}_k$（MSE 按对应步长组计算）。
- 校正器在 diag 上拟合，在 **cert** 上评估。
- **cert 结果对所有臂都不可见**，只写入 heldout 记录（A-M5）。因为在“一路 continue”的路径上，incumbent 序列与 cert 无关，所以跑满预算的运行可以**事后精确读出**认证停止点。不再需要单独的 CST profile。

### 2.5 反馈渲染（`feedback.py`）

四臂的渲染**共享一个外框**：标题、incumbent 的 diag MSE、按步长组的 MSE、历史候选表。差别只在中间一段：

| 臂 | 中间段内容 |
|---|---|
| `scalar` | 无 |
| `bj` | 1 步残差的 ACF（前 48 阶，按 lag 顺序）、Ljung–Box (Q,p)、周期图前 5 个峰 |
| `card` | 按 Bonferroni-LCB 排序的坐标增益（点估计 + 展示区间）、total 增益的 LCB、μ²、各坐标的 `by_horizon` 峰值区间 |
| `placebo` | 与 card 格式相同，但对坐标标签做**固定置换**（只保留这一种，A/C） |

- 措辞中性。卡片里的坐标名映射到所有臂共用的 `targetAspect` 词表（§3.3）。
- 记录每臂反馈的字符数和 token 估计，作为协变量。

### 2.6 候选契约与评估（`candidate.py`）

```python
# models/candidates/<id>/model.py —— 所有代码必须在这一个文件里
def create_adapter() -> Adapter

class Adapter:
    def fit(self, train: np.ndarray, meta: dict) -> None   # 只拿到 train（A-S1、B-S3）；早停自己在 train 内部切
    def predict_batch(self, X: np.ndarray) -> np.ndarray   # [B,L,C] → [B,H,C]
```

`meta` 包含 L、H、C、seed、时间戳频率，**不含数据集名称**。

**评估流程**：

1. **AST 白名单**（不是安全边界，只用来挡掉明显问题）：
   - 只允许 import `numpy`、`torch`、`math`、`scipy`；
   - 禁止 `open`、`os`、`sys`、`subprocess`、`socket`、`pathlib`、`pickle`、`importlib`、`eval`/`exec`，以及 `load/save/savez/tofile/fromfile/loadtxt/savetxt/genfromtxt/memmap`；
   - 允许随机数，这一点与 `_validate_deterministic_source` 不同。
   - 源码大小设上限，用来防止把 diag 值硬编码进代码。
2. **隔离子进程**（从 codex `_run_profile_process` 改写）：
   - `start_new_session` + killpg 超时回收；
   - **env 只给白名单**（PATH、线程数、CUDA/MPS 相关变量）；
   - cwd 设为临时目录；
   - 输入从父进程临时目录读入的 npz，结果写到文件，stdout 重定向到 stderr。
3. **只给 diag 和 cert 的输入窗口，不给目标**。test 由 posthoc 离线评估：对候选源码按相同 seed 重新执行 fit 和预测。
4. **交错分批**：同一批内各起点间距 ≥ L+H，从结构上杜绝批内转导（B 中-5）。另外**抽 50 个窗口逐个单独预测**，与批量结果比对（A-S2）。
5. **评估前后**对 workspace 文件树做哈希比对。
6. **超时**为 `candidateTimeoutSeconds`（默认 600）。因为不经过 MCP，没有 300s 上限。

**失败规则**：
- 语法错、import 被拒、smoke 失败：在同一会话里给 agent **最多 2 次修复回合**，并把错误信息回传。
- 仍然失败的候选记为 failed，**计入预算**。

---

## 3. 独立循环（`experiments/loop.py`）

### 3.1 一次 run 的流程

```
prepare(task, arm, seed):
  workspace = runs/<run>/ws          （agent 可见）
  heldout   = ~/.rhg-heldout/<random>/  （0700，位于 workspace 父目录树之外；路径只作为 Python 参数传递）
  写入 ws：data/series.npy（train+diag，通道去标识化）、meta.json、
          models/candidates/m0/model.py（模板）、models/incumbent/（m0 的副本）
  评估 m0 → incumbent 记录、首轮反馈、第 0 轮认证（结果写 heldout）

for round in 1..budget:
  runner = SdkRunner(SdkRunnerConfig(
      cwd=ws, system_prompt=SYSTEM, allowed_tools=["read","glob","grep","write","edit"],
      model/provider/env 取自 build_sdk_invocation_config(read_effective_llm_config(ws)),
      log=MessageLog(ws/logs/round-<k>.jsonl), session_role="main"))
  await runner.send(render_round(arm, incumbent, history, feedback))   # 每轮一个新会话
  校验：models/candidates/<id>/model.py 与 proposal.json 是否存在（缺失 → 修复回合，最多 2 次）
  smoke：在 train 的一小段上 fit/predict，只回传形状和错误，不回传任何指标（失败 → 修复回合）
  await runner.close()
  evaluate(candidate) → diag 残差 → DM 检验（HAC+HLN，单侧，α=0.05）→ accept/reject
  if accept: incumbent ← candidate；在 cert 上认证（只写 heldout）
  对 incumbent 生成新反馈；history.append(...)；追加写 runs/<run>/rounds.jsonl（断点续跑以轮为单位）
```

### 3.2 Prompt（`experiments/prompts/`）

- `system.md`：
  - 任务是改进一个多步预测器；
  - 只能在 `models/candidates/<new-id>/` 下写 `model.py` 和 `proposal.json`；
  - 基于 `models/incumbent/model.py` 修改；
  - 契约见上一节；
  - **不出现**数据集名称、领域描述，也不出现“残差”“频谱”等词。
- `round.md`：轮次、剩余预算、反馈块、历史表（所有臂格式相同），以及“请提出一个改进”。
- **四臂共用完全相同的 system/round 模板**，差别只来自反馈块。两份模板全文放进论文附录。

### 3.3 `proposal.json`（所有臂必填）

```json
{"candidateId": "c07", "targetAspect": "periodicity", "aspectDetail": "P≈24",
 "mechanism": "…", "expectation": "…"}
```

`targetAspect` 取值：`periodicity | long-range-beyond-window | cross-channel | horizon-range | level-bias | other`（A-M6，所有臂共用）。

### 3.4 记录

| 位置 | 内容 |
|---|---|
| `runs/<run>/rounds.jsonl`（agent 不可见，写在 ws 之外） | candidateId、parent、arm、round、diagMse、dm、accepted、proposal、源码指纹、耗时、反馈字符数、修复回合数、状态 |
| `heldout/records.jsonl` | **posthocCard**（每个候选都计算的完整卡片，与臂无关）、cert 认证结果、cert MSE |
| token | `usage.py` 汇总 `ws/.dsh-sessions/**/session.jsonl.zstd` 中 `assistant/message` 事件的 `data.usage`（inputTokens、outputTokens、cacheReadTokens、reasoningTokens；B 已在本地会话核对过）。reasoningTokens 是否已计入 outputTokens，要对照 DeepSeek 的计费口径再核实一次 |

### 3.5 隔离要点

- **runner 进程的环境变量里不放 heldout 路径**。`SdkRunner._runtime_env` 会复制 `os.environ`，这是 B-S5 指出的泄漏点。
- agent 没有 bash，无法执行 `env`、`ls ..` 这类探测，但 read/glob 不受路径限制。所以 posthoc **审计所有 read/glob/grep 调用中出现的、位于 ws 之外的路径**，一旦命中，该 run 作废。
- 消融“开放 bash + 可读 train/diag”放在 P4，单任务，用来回答“强 agent 会不会自己做诊断”（C）。

### 3.6 并行（`batch.py`）

- 预算按候选数计，所以各 run 可以并行，GPU 时间按进程分别记录。
- A800 服务器上 8–16 路并行，每路固定一张卡或一部分卡；本地 MPS 只跑冒烟测试。
- LLM 通过反向隧道访问（见 [REMOTE_SERVER.md](../REMOTE_SERVER.md)），**在 P1 做并发压测**。

---

## 4. 基线与离线分析

| 项 | 规格 |
|---|---|
| **一次性 m0+g** | total 线性校正器在完整 diag 上拟合，在 test 上评估（A-M7）。P4 再加一次性的非线性 g（随机特征 ridge） |
| **规则修复** | 按头号坐标选固定修法：band→对应周期的 Fourier 项；beyond-L→加长 lookback；channel→跨通道线性项；lag→AR 残差头。在 m0 模板上实现，不涉及 LLM，同样受预算约束 |
| **统一 m+g 后处理** | 对每个方法的最终 incumbent 都叠加一次在完整 diag 上拟合的 total 校正器，然后在 test 上评估 |
| auto.arima / ETS | 只在真实任务上跑（合成任务已有 oracle） |
| 不做 | TSFM、SEATS 复现、EvoCast+卡（P5） |

**`posthoc.py` 输出**：

- **主终点（预先登记）**：m+g 后处理之后、第 25 个候选时的 test 最优性差距（归一化），以及前 25 个候选的差距曲线下面积。
- **臂间检验**：以 run 为单位，做按任务分层的置换检验或 bootstrap（A-M10：不同 run 之间没有配对关系，不能用 DM 做臂间比较）。
- **缺陷定向率**：父模型的头号坐标在 diag 上选定；修改后它的 Δ（在 test 或新段上评估）降幅超过阈值，**并且**降幅大于其余坐标的平均降幅（A-M6）。按 `targetAspect` 与头号坐标是否对应分别统计。
- off-target 率、diag 与 test 增益的散点、token 和 GPU 时长。
- **认证事后停止点**；停止点之后继续搜索的收益（RQ3）。
- **审计**：ws 之外路径的访问、源码大小异常、diag 增益远大于 cert 增益的候选。
- **图 1**：横轴为候选数（另附 token 版本），纵轴为 test 最优性差距；叠加 card 臂在后台 cert 上的 headroom 区间，并换算到同一尺度；标出事后读出的认证停止点。

---

## 5. 阶段 0：验收门槛（`stage0.py`，必须先通过）

所有结果按（任务, 植入模型）逐格报告，不合并。植入缺陷和期望坐标的映射表**预先写死**：

| 植入缺陷 | 期望坐标 |
|---|---|
| 去掉季节项 | band(P) |
| 短 lookback（L=24） | beyond-L |
| 通道独立 | channel |
| 步长偏差 | horizon |

“粗 patch”找不到对应坐标，已删除（A-M11）。

| # | 检验 | 门槛 |
|---|---|---|
| G0-1 | **m = oracle 负对照**（每个任务） | LCB≤0 的覆盖率 ≥95%；判饱和率 ≥ 1−α |
| G0-2 | **区间双侧校准**：在新长段（10⁵ 步）上计算 $\Delta(\hat g)$ 的真值 | P(LCB≤Δ) 和 P(UCB≥Δ) 都约为 95%（ARFIMA 单独看） |
| G0-3 | **紧度**：对 gap ≥ 2ε·MSE 的植入缺陷 | Δ(ĝ)/gap 的中位数 ≥0.3；LCB 与 gap 的相关性 ≥0.8 |
| G0-4 | **定位**：头号坐标与预设映射一致 | ≥70% |
| G0-5 | **认证误停率**（对照 Δ(ĝ) 真值） | ≤α；同时给出“样本量—判饱和率”曲线，据此冻结 T、ε 和带宽 |
| G0-6 | **任务筛选**：一次性 m0+g 之后，剩余真实 gap 占初始 gap 的比例 | ≥30%（C 风险 1）。不达标的任务不用于 RQ2/RQ4，并重新设计缺陷：体制切换、乘性季节、长周期配合有限滞后等 |
| G0-7 | 观察 0 正对照：AR(1)、m≡0、h=1 | 实现增益 ≈ φ²s²；白噪声 ≈ 0 |

**判定**：G0-1、G0-2、G0-5、G0-7 任一不过，停下来修理论或实现；G0-6 不过，最多再花 1 周重新设计任务。

---

## 6. 测试清单

| 文件 | 内容 |
|---|---|
| `test_headroom_timeidx.py` | o 的约定；embargo；burn-in；截断不变性（全部坐标） |
| `test_headroom_synthetic.py` | Kalman 与解析解一致；HMM 联合递推与蒙特卡洛一致；ARFIMA 阶数 ≥ D；m=oracle 时 gap≈0 |
| `test_headroom_correctors.py` | 白噪声增益≈0；AR(1) 增益≈φ²σ²；band 用两个滞后时能恢复相位 |
| `test_headroom_inference.py` | 零假设下 DM-HAC-HLN 的第一类错误率 ≈α；解析置信区间的覆盖率 |
| `test_headroom_certify.py` | α 只在 incumbent 改变时消耗；Bonferroni 的数量；负对照饱和 |
| `test_headroom_candidate.py` | `fit` 拿不到 diag；**故意写的批内转导作弊候选被抓到**；子进程 env 里没有 heldout 路径和 key；候选 `print` 不破坏结果；超时由 killpg 回收；AST 拒绝文件 IO |
| `test_headroom_feedback.py` | 四臂外框一致；bj 不含 headroom 字段；placebo 是 card 的固定置换 |
| `test_headroom_usage.py` | zstd fixture 汇总 |
| `test_rhg_loop.py` | 用假 runner（不调用 LLM）跑 3 轮：断点续跑、修复回合、失败计入预算 |

现有 harness 代码不改，现有测试不受影响。

---

## 7. 阶段计划（今天 2026-10-15；ICML 约 2027-01-28；KDD 约 2027-02）

| 阶段 | 日期 | 交付 | 门槛 |
|---|---|---|---|
| **P0 核心与校准（M1）** | 10/16–10/22 | `headroom/` 全部核心模块、`stage0.py`、单测；**M1 三方代码评审** | **G0（10/22）**：§5 全部通过 |
| **P1 循环与基线（M2）** | 10/23–10/29 | `candidate`、`feedback`、`usage`、`loop.py`、prompts、m0 模板、一次性 m0+g、规则修复；**M2 三方代码评审** | **G1（10/29）**：card 臂无人值守 10 候选，失败率 <30%；8 路并发压测 |
| **P2 主实验（M3，冻结代码与 prompt，预先登记）** | 10/30–11/12 | 3 合成 + 1 真实（去标识化）× 3 臂 × **8 run** × 25 候选；`batch`/`posthoc`/`figures`；全部基线；统一 m+g 后处理；Exchange-Rate 负对照；**M3 三方代码评审** | 主终点分层置换检验 p<0.05，且 m+g 后处理后仍成立 |
| **P3 补充（可砍）** | 11/13–11/30 | Placebo；换 LLM；开放 bash 消融；非线性 g 配 bilinear/TAR；GIFT 子集；H=96 | 按结果取舍 |
| 缓冲与出图 | 12/1–12/15 | 重跑失败格、图 1 定稿 | — |
| 写作 | 12/16–1/25 | ICML 投稿（KDD 备选） | — |
| **P5（投稿后）** | 2–3 月 | harness 集成：RHG 变体、MCP 工具、overlay；按评审 B 的“最小集成路径”实施；ECG 领域坐标复用 KTL；“EvoCast + 卡” | 不进论文主线 |

**成本估算**（DeepSeek v4-flash，约 70k token/候选）：P2 约 19M token，P3 约 170M token，合计约 US$100–300。换用其他 LLM 的消融需要单独预算。

---

## 8. 需要同步回金文档 v0.2 的修改

1. H 从“96–720”改为“主实验 H=24，补充 H=96”；信息集写明 lag 相对起点，并包含同步长的已实现残差。
2. §3.3 卡片示例删掉“certificate: UCB on cert”，改为说明“认证结果对 agent 不可见，停止点由后台判定”。
3. §3.5 认证只在 incumbent 改变时检验；α 均匀分配为 α/J_max；统计量集合为 {total, 各坐标, 步长组}。
4. §3.4 写明 agent 没有 bash，评估由循环在 agent 之外执行；“开放 bash”作为消融。
5. §4.2 oracle 定义为“信息深度 ≥ D 的全信息最优”；合成任务的 cert/test 用独立新模拟段。
6. §4.4 统计协议：每格 8 run；以 run 为单位做分层置换检验；预先登记主终点。
7. §4.6 / §5：EvoCast+卡、领域坐标移到未来工作或附录。

## 9. 留给作者确认

1. **D1（独立循环，harness 集成推迟到 P5）是本计划与初版差别最大的决定**。如果你希望论文里明确体现“基于 RecursiveECG 构建”，可以在 P3 之后把 loop 迁移进 harness 作为演示，主实验数据仍来自独立循环。
2. **服务器并行**依赖反向隧道承受 8–16 路 LLM 并发，要在 P1 压测。如果撑不住，需要考虑在服务器上直接部署可访问的 LLM 端点。
3. G0-6 任务筛选不过时，需要重新设计合成缺陷，这是关键路径上最大的不确定性。
