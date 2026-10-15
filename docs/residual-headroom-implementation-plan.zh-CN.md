# Residual Headroom 实验系统：编码方案（初版，已被取代）

> **已被 [最终编码计划](residual-headroom-final-coding-plan.zh-CN.md) 取代**，本文仅作评审记录保留。

> 金文档：[minipaper v0.2](minipaper-residual-guided-discovery-v0.2.zh-CN.md)。本方案只为论文实验服务，原则是**能复用就复用，不做产品化**。
> 日期：2026-10-15。代码引用中的 `codex:` 前缀表示 `codex/profile-optimization-weather` 分支上的文件。

---

## 0. 总体决策

### 0.1 起点分支

**从 `codex/profile-optimization-weather` 新开 `research/residual-headroom`。**

- 该分支 = `dsh-backend-develop` + 40 个 PFO 提交（`git log dsh-backend-develop..codex/...` 无反向差异），是严格超集。
- 它已经有我们需要的预测基础设施：通用数值 CSV 加载、`ForecastAdapter` 协议、`models/candidates/<id>/model.py` + `create_adapter()` 动态加载、可回收进程组的超时子进程、一次性 holdout、线性预测 adapter、PFO 变体接入 MCP 的完整样板。
- 硬件 profiling（`profiling/`、`bootstrap.py`、`constraints.py`、`dashboard.py`）不删也不用，新模式不依赖它们。

### 0.2 分层

```
┌────────────────────────────────────────────────────────────────┐
│ 实验层 experiments/   规格、批量运行、分析与出图（离线脚本）   │
├────────────────────────────────────────────────────────────────┤
│ 集成层 已有 harness   变体 RHG、2 个 MCP 工具、节点 overlay     │
│ (DSH 会话/恢复/日志/节点链 全部原样复用)                        │
├────────────────────────────────────────────────────────────────┤
│ 核心层 backend/harnessing_ts/headroom/   纯 numpy/scipy、确定性 │
│ 合成任务+oracle · 残差张量 · 校正器 · 统计推断 · 缺陷卡 · 认证  │
└────────────────────────────────────────────────────────────────┘
```

- **核心层不依赖 LLM，也不依赖 harness**，可以单独测试。阶段 0（证书校准）只用这一层。
- **LLM 只能调用核心层，不能修改它。** 这一点沿用现有的“后端确定性实现、agent 只能调用”原则（与 `reference_feature_extractor` 的校验器、PFO 的冻结 profile 工具同一思路）。

### 0.3 刻意不做的事

- 不新增节点类型：复用 `problem-contract`（由后端预填）→ `iterative-solving` ×N → `final-summary`。
- 不改前端：缺陷卡写成 markdown，用现有的文件预览查看。
- 不接知识图谱和 KTL。领域坐标放到 P4，并且是可选的。
- 不使用 SUB 子代理：单父代、每轮一个候选。

---

## 1. 复用清单

| 已有组件 | 位置 | 复用方式 |
|---|---|---|
| DSH 会话、暂停恢复、日志、SSE | `agent/`、`execution.py`、`api/` | 原样使用 |
| 节点链与 `loopDecision` 路由 | `node_state.py` | 原样使用；RHG 模式下额外校验 `loopDecision` 必须与后端的循环决策一致（§4.4） |
| 变体系统 | `variants/profiles.py` | 新增特征码 `RHG` 与反馈臂 `FBS/FBJ/FBC/FBP`（§4.1） |
| 节点 prompt overlay | `variants/prompts/*.md`、`AblationVariant.node_prompts` | 新增 `rhg-iterative.md`、`rhg-final.md`；**四个臂共用同一份 overlay** |
| 工具白名单 | `tools/compose_tools.py` | 新增 RHG 工具池（照搬 codex 分支的 PFO 写法） |
| MCP 桥与 capability/lease 防护 | `mcp/server.py`、`orchestrator.py` | 新增两个工具 `get_feedback`、`evaluate_candidate`；走现有 `_call_control` 与 `/internal/dsh/mcp` |
| `record_run` / timeline | `workspace_store.py` | 每次候选评估写一条 run 记录 |
| runtime settings | `workspace_store.read_runtime_settings` | 新增 `candidateBudget`（同一校验写法） |
| 预测 CSV 加载 | `codex:optimization/weather.py::load_forecast_csv` | 原样使用 |
| 模型协议与动态加载 | `codex:optimization/models.py::ForecastAdapter`、`candidates.py::_load_candidate_adapter` | 泛化成 `headroom/candidates.py`：去掉 `profile_callback`/`save` 要求，增加源码检查 |
| 超时子进程 | `codex:optimization/evidence.py::_run_profile_process` | 原样复用（killpg 回收进程组） |
| 一次性 holdout | `codex:optimization/holdout.py` 的思路 | test 只由离线分析脚本读取，agent 和后端循环都不读取 |
| 线性预测 adapter | `codex:optimization/linear_forecast.py`、`cross_channel_forecast.py` | 作为阶段 0 的“植入缺陷模型”和 $m_0$ 初始父模型 |
| 源码安全检查 | `reference_feature_extractor.py::_validate_deterministic_source` | 借鉴其写法，禁止候选代码做文件和网络 IO（§3.3） |
| 自适应停止 ADA | 变体能力 | 停止条件改为“预算耗尽或认证饱和”，由后端判定 |
| dry-run | `TS_HARNESS_DRY_RUN` | 集成测试使用 |
| optional torch 依赖 | `codex:pyproject.toml` | 候选模型可以用 torch（MPS） |

---

## 2. 核心层 `backend/harnessing_ts/headroom/`

预计约 1500 行。依赖：`numpy`、`scipy`（新增 optional 组 `research`）。

```
headroom/
├── __init__.py
├── config.py        # HeadroomConfig（照抄 codex config.py 的 dataclass + validate + camelCase 风格）
├── splits.py        # 时间切分与滚动起点窗口
├── synthetic.py     # 合成过程 + 精确最优预测器（oracle）
├── residuals.py     # 残差张量与“已实现信息集”特征
├── correctors.py    # 按坐标的线性/非线性校正器，以及 held-out 实现增益
├── inference.py     # block bootstrap、DM+HAC、置信界、alpha-spending
├── spectral.py      # KS 闭式、multitaper、IS 密度（仅可视化）
├── feedback.py      # 四个反馈臂的渲染：Scalar / BJ / Card / Placebo
├── certify.py       # ε-饱和停止规则
├── candidates.py    # 候选加载、源码检查、子进程评估
└── service.py       # 供 orchestrator 调用的门面：evaluate / feedback / loop 决策
```

### 2.1 `config.py`：`config.headroom.json`

```json
{
  "schemaVersion": 1,
  "task": {"inputLength": 96, "horizon": 96, "featureMode": "M", "targetColumn": null},
  "splits": {"train": 0.6, "diag": 0.2, "cert": 0.1, "test": 0.1, "diagASplit": 0.5},
  "arm": "card",
  "budget": {"candidates": 30, "candidateTimeoutSeconds": 900},
  "acceptance": {"alpha": 0.05},
  "certify": {"alpha": 0.05, "epsilon": 0.02, "spending": "geometric"},
  "coordinates": {
    "bands": "auto",
    "lagSegments": [[1, 24], [25, 96], [97, 336]],
    "maxChannelPairs": 16,
    "horizonGroups": [[1, 24], [25, 96]]
  },
  "corrector": {"ridgeAlphas": [0.1, 1, 10, 100], "nonlinear": false},
  "inference": {"bootstrap": 500, "blockLength": "auto"}
}
```

- 写法照搬 codex `OptimizationConfig`：frozen dataclass、`validate()` 汇总全部错误、加载时 fail fast。
- `bands: "auto"` 的含义：对数等分频段，再加上从 diag 周期图检出的前 3 个周期峰（固定规则）。

### 2.2 `splits.py`

```python
@dataclass(frozen=True)
class SplitIndex:
    train: slice; diag_a: slice; diag_b: slice; cert: slice; test: slice

def make_splits(n: int, cfg: HeadroomConfig) -> SplitIndex
def origins(split: slice, L: int, H: int) -> np.ndarray   # 目标完全落在 split 内的起点 o
def windows(values, origins, L, H) -> tuple[np.ndarray, np.ndarray]  # X[B,L,C]，Y[B,H,C]
```

- 标准化统计量只用 train 计算，与 codex `normalized_weather` 的做法一致。

### 2.3 `synthetic.py`：最优已知任务

```python
@dataclass
class SyntheticTask:
    name: str
    values: np.ndarray                 # [T, C]
    timestamps: tuple[str, ...]
    oracle_forecast: Callable[[int], np.ndarray]   # 起点 o → [H, C]，基于 I_o 的精确最优预测
    oracle_mse: np.ndarray             # [H]，按步长的理论最优 MSE（可得时）
    planted: dict                      # 植入的结构，供阶段 0 对照

def lgssm(spec) -> SyntheticTask       # 多周期季节 + 部分观测 + 跨通道滞后耦合；Kalman 滤波 + h 步外推
def hmm_switching(spec) -> SyntheticTask   # 前向滤波，按后验混合各体制的预测
def arfima(spec) -> SyntheticTask      # 有限 lookback 的最优用 Durbin–Levinson 计算
def bilinear(spec) / tar(spec)         # 白噪声但非线性可预测；oracle 为条件期望（可闭式或蒙特卡洛）
def garch(spec) / random_walk(spec)    # 负对照：均值不可预测，oracle 分别为 0 与“最后一个值”
REGISTRY: dict[str, Callable]          # 名称 → 生成器，供实验规格引用
```

- oracle 同时给出两种参照（对应论文 §4.2）：(i) 信息集 $\mathcal{I}$ 下的最优；(ii) 正确设定的参数模型在同一训练集上拟合（`fit_reference_model`）。
- **oracle 与 cert/test 数据只写在实验目录，不进入 workspace**（§5.3）。

### 2.4 `residuals.py`

```python
@dataclass
class ResidualTensor:
    origins: np.ndarray          # [B]
    r: np.ndarray                # [B, H, C]，r[o,h] = x_{o+h} - x̂^{(h)}_o

def residuals(pred: np.ndarray, y: np.ndarray, origins) -> ResidualTensor

def realized_features(rt: ResidualTensor, raw: np.ndarray, o: int, *, coord: Coordinate) -> np.ndarray
    """只返回在起点 o 已实现的信息：对目标时刻 t<=o，取 r[t-j, j]（j>=1，已实现的跨步长残差），
    以及 raw[s] (s<=o)。各坐标只是选取其中的不同子集。"""
```

**信息集的约定**（论文 §3.1）：目标时刻 $t\le o$ 的“最新已实现残差” $e_t=r[t-1,1]$，加上其余已实现的步长 $r[t-j,j]$；原始观测可以超出 lookback $L$。实现上统一构造一张 $e_t$ 时间序列表，再按坐标取滞后，避免逐起点重复计算。

### 2.5 `correctors.py`

```python
@dataclass(frozen=True)
class Coordinate:
    kind: Literal["total", "band", "lag", "channel", "horizon", "domain"]
    spec: dict                     # 如 {"period": 24, "lo": ..., "hi": ...} / {"k1":97,"k2":336} / {"src":2,"dst":5}
    label: str

def enumerate_coordinates(cfg, diag_periodogram) -> list[Coordinate]

@dataclass
class Gain:
    coord: Coordinate
    delta: float                   # held-out 实现增益（MSE 降幅）
    lcb: float; ucb: float
    by_horizon: np.ndarray         # [H]
    mu2: float                     # 均值偏移项，单独报告

def fit_corrector(rt_a, raw, coord, cfg) -> FittedCorrector   # ridge；alpha 只在 A 内做时间序列 CV
def realized_gain(fc, rt_b, raw, inference_cfg) -> Gain        # 在 B（或 cert）上评估，块 bootstrap 给置信界
```

- **band**：对 $e_t$ 做**因果** FIR 带通（`scipy.signal.firwin`，单侧卷积），取若干滞后作为特征。
- **lag**：直接取 $e_{o-k}$（$k\in[k_1,k_2]$）。$k>L$ 时额外加入 $x_{o-k}$，即“lookback 外信息”坐标。
- **channel**：通道 $j$ 的 $e$ 与 $x$ 的过去，用来预测通道 $i$ 的残差。先用多变量 Szegő 的 log-det 估计总量，只有显著时才展开到前 `maxChannelPairs` 个通道对。
- **total**：以上全部特征的并集，作为认证与“总 headroom”的统计量。
- `nonlinear=true` 时换成两层 MLP（sklearn 或 numpy 均可），仅用于消融。

### 2.6 `inference.py`

```python
def moving_block_bootstrap(d: np.ndarray, block: int, n: int, rng) -> np.ndarray  # d 为逐起点的损失差
def auto_block_length(d, H) -> int          # max(H, Politis–White 估计)
def dm_test(loss_a, loss_b, H) -> tuple[float, float]   # Newey–West HAC，滞后 = H-1；返回单侧统计量和 p 值
def lcb_ucb(d, level, block) -> tuple[float, float]
def alpha_schedule(alpha, k, kind="geometric") -> float # alpha_k = alpha·2^{-k}，Σ ≤ alpha
```

- 损失差按起点聚合：$d_o=\frac{1}{HC}\sum_{h,c}\big[r^2-(r-\hat g)^2\big]$，再在这个序列上做 bootstrap 或 DM 检验。

### 2.7 `spectral.py`（仅用于可视化和 BJ 臂）

```python
def multitaper_psd(e, K=5) -> tuple[freqs, psd]              # 剔除 ω=0 和 ω=π
def ks_innovation_variance(psd, K) -> float                  # exp(mean log psd + (ψ(K) - log K)) 偏差修正
def is_density(psd, sigma2_ks) -> np.ndarray                 # σ²(x-1-log x)
def acf(e, max_lag); ljung_box(e, lags) -> list[(lag, Q, p)]
```

### 2.8 `feedback.py`：四个反馈臂

所有臂都返回同一个结构 `{"text": str, "payload": dict, "tokens_estimate": int}`，**只是内容不同**：

| 臂 | 内容 |
|---|---|
| `scalar` | 当前 incumbent 的 diag MSE、按步长的 MSE，以及历次候选的 (id, MSE, accepted) |
| `bj` | scalar 的内容 + 1 步残差的 ACF（前 48 阶）、Ljung–Box (Q, p)、周期图前 5 个峰。**按 lag/频率顺序排列，不排序、不给 headroom、不给认证** |
| `card` | scalar 的内容 + 缺陷卡：按 LCB 排序的坐标增益、总 headroom 的 LCB、μ² 项、认证状态 |
| `placebo` | 与 card 格式完全相同，但内容取自另一个任务、另一个模型的卡片（固定映射表），或把坐标标签随机打乱。实验规格里二选一 |

缺陷卡的 markdown 模板与论文 §3.3 示例一致。同时写入 `reports/headroom/<candidate-id>-feedback.md`，供前端文件预览查看。

### 2.9 `certify.py`

```python
@dataclass
class CertDecision:
    round: int; ucb: float; threshold: float; alpha_k: float; saturated: bool

def certify(incumbent_rt_diag, incumbent_rt_cert, raw, cfg, k) -> CertDecision
    """total 校正器在 diag 上拟合，在 cert 上算实现增益的 UCB（置信水平 1-alpha_k）；
    UCB < epsilon·MSE_cert 即判定饱和。"""
```

- cert 的结果只向循环返回一个比特（停 / 不停）。即使是 card 臂，agent 也只看到 saturated 这一个标志，看不到 cert 上的数值。

### 2.10 `candidates.py`：候选契约与评估

**候选目录**沿用 codex 约定，写在 workspace 中：

```
models/candidates/<candidate-id>/
├── model.py        # 必须导出 create_adapter()
├── README.md       # 由 agent 写：目标坐标、机制假设、预期
└── (agent 自由的其他文件)
```

**Adapter 协议**（从 codex `ForecastAdapter` 精简而来）：

```python
class ForecastAdapter(Protocol):
    model_id: str
    def fit(self, train: np.ndarray, diag: np.ndarray, meta: dict) -> None: ...
    def predict_batch(self, history: np.ndarray) -> np.ndarray: ...   # [B,L,C] → [B,H,C]
```

`meta` 中包含 L、H、通道数、时间戳和 seed，**不包含数据集名称**。

**评估流程**（`evaluate_candidate_process`）：

1. **源码检查**：AST 层面禁止 `open`、`os`、`pathlib`、`subprocess`、`socket`、`urllib`、`requests`、`pickle`、`shutil`、`importlib` 等（借鉴 `_validate_deterministic_source`）；允许 `numpy`、`torch`、`math`、`scipy`。
2. **子进程**：复用 `_run_profile_process`，带超时，并用 killpg 回收进程组。入口为 `python -m harnessing_ts.headroom.candidates --workspace W --candidate ID --inputs <npz>`。
3. **输入**：父进程预先把 train/diag 原始值，以及 diag/cert/test **所有起点的历史窗口 X**（不含目标 Y）写入一个临时 npz。子进程只能拿到这些数据。
4. **按时间顺序分批预测**；之后再用一个打乱的批次顺序重跑一小部分并比对结果，检测模型是否在窗口之间缓存信息作弊。
5. **输出**：预测写在实验目录的 `preds/<id>.npz`（diag/cert/test），workspace 里只留 diag 的预测。
6. 父进程计算残差、做 DM 检验、决定是否接受、生成反馈、做认证判定，并写入记录。

---

## 3. 服务门面 `headroom/service.py`

```python
class HeadroomService:
    def __init__(self, workspace: Path, heldout_dir: Path, cfg: HeadroomConfig, arm: str): ...
    def initialize(self, m0_id: str) -> dict          # 评估初始父模型 m0，写 incumbent，生成首张反馈
    def get_feedback(self) -> dict                    # 当前 incumbent 在本臂下的反馈
    def evaluate(self, candidate_id, target_coord, expectation) -> dict
    def loop_decision(self) -> dict                   # {"next": "continue"|"exit", "reason": "budget"|"saturated"|...}
```

`evaluate` 返回给 agent 的字段（所有臂相同）：`accepted`、`diagMse`、`dmPValue`、`reason`、`budgetRemaining`，以及按本臂渲染的新反馈。

`evaluate` 同时写入 `state/headroom/evaluations.jsonl`（agent 不可见的字段写到实验目录的同名镜像中），每条包含：

| 字段 | 含义 |
|---|---|
| candidateId, parentId, arm, round | 基本信息 |
| diagMse, dm, accepted | 接受判定 |
| claimedTarget, expectation | agent 声称的目标坐标和预期 |
| **posthocCard** | 对每个候选都计算的完整缺陷卡，与臂无关 |
| **directed** | 修改是否缩小了父模型卡片上的头号缺陷（缺陷定向率） |
| **offTarget** | MSE 下降但 claimedTarget 没有缩小 |
| **cert** | CertDecision |
| sourceFingerprint, seconds, feedbackTokens | 可复现与成本统计 |

加粗字段只写到实验目录，用于离线分析（对应论文 §4.4 的指标）。

---

## 4. 集成层改动

### 4.1 变体 `variants/profiles.py`

新增特征码：

| 码 | 含义 |
|---|---|
| `RHG` | residual-headroom 研究模式（独立模式，和 PFO 一样不能与 KGR/RQA/FQA 组合） |
| `FBS` / `FBJ` / `FBC` / `FBP` | 反馈臂：scalar / 原始 BJ / 缺陷卡 / 安慰剂。四者互斥，RHG 必须且只能选一个 |
| `CST` | 认证停止生效（不加 CST 时只记录认证结果，跑满预算） |

注册的 profile：`NOD-RHG-FBS`、`NOD-RHG-FBJ`、`NOD-RHG-FBC`、`NOD-RHG-FBC-CST`、`NOD-RHG-FBP`。

- 主实验 RQ2 中四个臂都**跑满预算**，从同一次运行事后读出“认证停止点”和“停止后继续搜索的收益”（RQ3）。`FBC-CST` 只用于演示真实停止。
- 所有 RHG profile：`knowledge_to_tools=False`、`independent_subagents=False`、`case_review=False`；`node_prompts` 统一指向 `rhg-iterative.md` 和 `rhg-final.md`。

### 4.2 工具 `tools/compose_tools.py`、`mcp/server.py`

```python
RHG_TOOLS = ["mcp__ts_harness__get_feedback", "mcp__ts_harness__evaluate_candidate"]
```

只在 RHG 变体的 `iterative-solving` 节点中加入。

MCP 注册（沿用现有的 `intend` 必填参数和 `_call_control`）：

```python
@server.tool(name="get_feedback")
async def get_feedback(intend: str) -> str

@server.tool(name="evaluate_candidate")
async def evaluate_candidate(candidateId: str, targetCoordinate: str, expectation: str, intend: str) -> str
```

`targetCoordinate` 和 `expectation` 对**所有臂**都是必填的（论文 §3.4：所有组都要写预期）。

### 4.3 `orchestrator.py`

- 新增 `request_get_feedback` 和 `request_evaluate_candidate` 两个方法，直接委托给 `HeadroomService`。现有的 lease/epoch 防护自动生效。
- `evaluate` 是阻塞的子进程调用，用 `asyncio.to_thread` 包装，与现有的 `request_validate_reference_feature_extractor` 一致。
- `initialize()` 在 RHG 模式下额外执行三件事：
  1. 根据 `config.headroom.json` 生成 `user/problem-contract.md` 和 `user/data-spec.md`（模板化，跳过 problem-contract 节点）；
  2. 调用 `HeadroomService.initialize(m0)`；
  3. 发布 timeline 事件 `headroom_initialized`。

### 4.4 `node_state.py`

在 RHG 模式下，`validate_finish_control` 增加一条规则：`loopDecision` 必须等于 `HeadroomService.loop_decision()["next"]`，否则拒绝 `finish_node`。停止由后端决定，LLM 不能自行提前结束或无限继续。

### 4.5 prompt overlay `variants/prompts/rhg-iterative.md`（要点）

1. 先调用 `get_feedback`，读取 `user/iteration-state.md` 和历史报告。
2. 只基于父模型 `models/incumbent` 的源码写**一个**新候选 `models/candidates/<id>/model.py`，并在 README 中写明目标坐标和预期。
3. 调用 `evaluate_candidate`，结果以后端返回为准。
4. 写 `reports/iterations/<id>-summary.md` 和 `user/iteration-state.md`。
5. `finish_node` 时，`loopDecision` 照抄后端返回的 `next`。

overlay 中**不出现**数据集名称、领域描述、“残差”“频谱”等可能引导某一臂的词。四个臂的差异只来自工具输出。

### 4.6 workspace 布局（`state/workspace_layout.py`）

```
data/raw/series.csv            # 只含 train+diag；列名去标识化为 ch_0…；时间戳保留
config.headroom.json
models/incumbent -> candidates/<id>   # 由后端维护（复制或符号链接）
models/candidates/<id>/
reports/headroom/*.md
state/headroom/evaluations.jsonl      # agent 可见的精简版
```

---

## 5. 实验层 `experiments/`

```
experiments/
├── specs/                      # 实验规格（JSON）
│   ├── stage0_calibration.json
│   ├── stage1_factorial.json
│   └── ...
├── run.py                      # 批量运行：任务 × 臂 × seed → 每次一个独立 workspace
├── stage0.py                   # 证书校准，不涉及 LLM
├── baselines.py                # m0+g boosting、规则修复、auto.arima/ETS、TSFM
├── posthoc.py                  # 离线读取 test 和 oracle，计算全部指标
└── figures.py                  # 主图 1 与其余图
```

### 5.1 阶段 0：`stage0.py`（**最先做**）

```
for task in [lgssm_multiperiod, lgssm_xchannel, arfima, bilinear, garch, random_walk]:
    for model in planted_defect_models(task):   # 短 lookback / 通道独立 / 粗 patch / 无季节项（线性 adapter 加不同限制）
        rt = residuals(model, task)
        gains = [realized_gain(fit_corrector(...), ...) for coord in coords]
        true_gap = mse(model) - oracle_mse(task)
        record(lcb_total <= true_gap, top_coord == planted_defect, saturated_at_round0)
```

**验收标准**（对应论文 §7）：
- LCB ≤ 真实差距的覆盖率 ≥ 95%（多 seed 统计）；
- 头号坐标与植入缺陷的一致率明显高于随机；
- GARCH / 随机游走在第 0 轮判定饱和的比例 ≥ 1−α。

### 5.2 批量运行 `run.py`

- 为每个 (task, arm, seed) 新建 workspace，写入 CSV、config 和 m0；设置 `TS_HARNESS_VARIANT=NOD-RHG-<arm>`、`TS_HARNESS_CONTROL_MODE=auto`。
- 新增 CLI 子命令 `ts-harness run-rhg --workspace W --heldout-dir D`：启动 orchestrator，直接 `request_enter_node("iterative-solving")`，然后等待 `is_pipeline_complete` 或失败。这复用现有 CLI 的 `start-node` 路径，并补一个“等待完成”的循环。
- 同一机器上多次运行**串行**执行（GPU/MPS 计时与资源互不干扰）；不同 workspace 完全隔离。

### 5.3 数据隔离

- cert/test 的原始数据、oracle、完整评估记录只放在 `experiments/runs/<run>/heldout/`，通过仅限服务端的环境变量 `TS_HARNESS_HEADROOM_HELDOUT_DIR` 传入。prompt 和 workspace 文件中都不出现这个路径。
- 候选子进程只拿到“没有目标值”的历史窗口（§2.10）。
- `posthoc.py` 会扫描 DSH 会话日志，检查是否出现过 heldout 路径。一旦出现，该次运行作废。

### 5.4 基线 `baselines.py`

| 基线 | 实现 |
|---|---|
| $m_0+g$ 迭代 boosting | 直接调用 `correctors.py`：每轮拟合 total 校正器并叠加，重复到预算耗尽或饱和。不涉及 LLM |
| 规则修复 | 头号坐标 → 固定修法（band：加对应周期的 Fourier 特征；lag>L：加长 lookback；channel：加跨通道线性项），在线性 adapter 上实现 |
| auto.arima / ETS / Theta | `statsforecast`（加入 optional 组 `research`），逐通道拟合 |
| 零样本 TSFM | Chronos-2 / TimesFM，独立脚本，放在 P3 |
| **统一 $m+g$ 后处理** | `posthoc.py` 对每个方法的最终模型都叠加一次 total 校正器，然后再比较（论文 §4.3） |

### 5.5 离线指标 `posthoc.py` 与图 `figures.py`

- 最优性差距随候选数、GPU 时长、token 的曲线（合成任务）。
- 证书校准散点：LCB 与真实差距；误停率与漏停率。
- 缺陷定向率、off-target 率（覆盖所有臂的全部提案）。
- diag 增益与 test 增益的散点。
- 统计：每组 5–10 次运行，报告 IQR；组间比较用配对检验。
- **图 1**：一个合成任务上各臂的最优性差距曲线，叠加 card 臂估计的 headroom 置信带，并标出认证停止点。

---

## 6. 测试计划（`tests/`）

| 测试文件 | 覆盖内容 |
|---|---|
| `test_headroom_synthetic.py` | oracle 正确性：对已知过程，经验 MSE 收敛到理论值；Kalman 预测与解析解一致 |
| `test_headroom_correctors.py` | 白噪声残差的实现增益约为 0；植入 AR(1) 残差时增益约为 φ²σ²；校正器只使用已实现信息（构造“偷看未来”特征时测试应报错） |
| `test_headroom_inference.py` | DM 检验在零假设下的第一类错误率约为 α；bootstrap 置信区间覆盖率 |
| `test_headroom_certify.py` | 负对照饱和；植入缺陷时不饱和；alpha 序列的总和不超过 α |
| `test_headroom_candidates.py` | 源码检查拒绝 IO 调用；子进程超时会被回收；打乱批次顺序时检测出跨窗口作弊 |
| `test_headroom_feedback.py` | 四个臂的输出结构一致；placebo 与 card 格式相同；bj 中不出现 headroom 字段 |
| `test_rhg_variant.py` | 变体解析、互斥规则、工具白名单（仿照 `test_ablation_variants.py`） |
| `test_rhg_loop_control.py` | `loopDecision` 与后端决策不一致时被拒绝（仿照 `test_node_control_flow.py`） |
| `test_rhg_mcp.py` | 两个工具经过 capability/lease 防护（仿照 `test_dsh_mcp_control.py`） |

此外，现有测试全部不能回归。

---

## 7. 分阶段交付

| 阶段 | 内容 | 验收 | 估计 |
|---|---|---|---|
| **P0 核心与证书校准** | `config/splits/synthetic/residuals/correctors/inference/spectral/certify` + `stage0.py` + 对应单测 | §5.1 的三条验收标准；**不通过就不进入 P1** | 5–7 天 |
| **P1 循环接入** | `candidates/feedback/service` + 变体 + 两个 MCP 工具 + node_state 规则 + overlay + `run-rhg` | dry-run 下完整跑通 3 轮；真实 LLM 在 1 个合成任务上跑通 card 臂 10 轮 | 5–7 天 |
| **P2 最小验证实验** | `run.py` / `baselines.py`（$m+g$、规则修复）/ `posthoc.py` / `figures.py`；Scalar / BJ / Card 三臂 × 2 个合成任务 + 1 个真实任务 × 30 候选 × 5 次运行 | 论文 §7 阶段 1 的两个成功信号 | 5 天，外加运行时间 |
| **P3 完整实验** | Placebo 臂、全部合成族、GIFT-Eval 子集（去标识化）、auto.arima/ETS、TSFM、非线性校正器与其余消融、换 LLM | 论文 §4 的全部 RQ | 1–2 周，外加运行时间 |
| **P4 可选** | 领域坐标（复用 KTL 的“生成 → 后端校验”）、“EvoCast + 缺陷卡”插件实验、附录中的逐层定位 | 视 P3 的结果决定 | — |

---

## 8. 需要确认或核实的事项

1. **起点分支**：确认是否从 `codex/profile-optimization-weather` 开 `research/residual-headroom`。该分支尚未推送，前端也有拟态视觉改动，但不影响本方案。
2. **token 计量**：DSH 会话日志里是否记录了 LLM 的 usage 需要核实（目前只在 `translate.py` 里看到 `*_tokens` 类遥测）。如果没有，退而用 prompt 和工具输出的字符数估算，并在论文中注明。
3. **候选代码隔离**：AST 检查加“只给历史窗口”已能防止绝大多数泄漏，但 agent 有 bash 权限。若需要更强的隔离，可以用独立的系统用户运行 DSH 运行时。建议先用日志审计，P3 再视情况加强。
4. **`statsforecast` / `scipy` 依赖**：放在 optional 组 `research` 里，不影响主服务的安装。
