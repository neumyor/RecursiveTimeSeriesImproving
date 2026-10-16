# headroom 包接口契约

> 本文档对全部实现具有约束力。任何与本文冲突的实现以本文为准。
> 总体计划见 [residual-headroom-final-coding-plan](residual-headroom-final-coding-plan.zh-CN.md)。

语言与依赖：Python 3.11+，核心层只用 `numpy`、`scipy`（`usage.py` 另需 `zstandard`）。禁止 import harness（orchestrator/DSH/torch）。测试运行方式：仓库根目录 `uv run --extra research pytest tests/<file> -q`。

## 1. 全局约定

- 数值数组一律 `float64`。`values: [T, C]`，通道即列，下标即时间。
- **起点 o**：输入窗口最后一个观测的下标。输入 `x[o-L+1..o]`，目标 `x[o+1..o+H]`，`h = 1..H`。
- **残差** `r[i, h-1, c] = x[o_i+h] - pred[i, h-1, c]`。
- **e 序列**：`e_full[t] = r_full[t-1, 0, :]`——在起点 t-1 做出、目标恰为 x[t] 的 1 步残差（即 e_t = r[t-1,1]）。
  这是「在时刻 t 已实现」的最新 1 步残差；非 NaN 行为 `[seg_start+1, seg_stop-H]`。
  （注意：不得写成 r[origin t, h=1]——那会让 lag-0 特征等于 h=1 目标本身，造成目标泄漏。）
- **段（segment）**：一段含前缀历史的连续数组（局部下标从 0 开始）。段的可用起点 `o` 满足 `o >= burn_in` 且 `o + H <= T - 1`；`burn_in = max_lag + fir_len_cap`（见 config）。这保证起点 o 的全部特征只依赖段内已实现信息。
- 标准化统计量只来自 train。
- **增益符号**：`delta = MSE_m - MSE_{m+g}`（在评估段上），正值表示改进。
- 一切随机性显式接受 `np.random.Generator` 或 `seed`；同 seed 完全可复现。
- 输入校验抛 `ValueError`；运行时/子进程失败抛 `RuntimeError`（或子类），消息含期望与实际。不做多余防御，模块间信任本契约。

## 2. `headroom/config.py`

```python
@dataclass(frozen=True)
class HeadroomConfig:
    input_length: int = 96                  # L
    horizon: int = 24                       # H（主实验 H=24）
    evolution: bool = True                  # 多样性裁决（演化引导），三臂共用；arm 是循环参数，不是配置字段
    budget_candidates: int = 25
    candidate_timeout_seconds: float = 600.0
    acceptance_alpha: float = 0.05          # DM 接受检验
    certify_alpha: float = 0.05              # 总误停预算
    certify_epsilon: float = 0.02            # ε，相对 cert MSE
    lag_segments: tuple[tuple[int, int], ...] = ((0, 23), (24, 95), (96, 335))
    horizon_realized_lags: int = 4          # r[o-h-k, h]，k=0..3
    max_period_peaks: int = 3
    band_period_range: tuple[int, int] = (4, 400)
    fir_len_cap: int = 256
    beyond_l_lags: int = 10                  # k = L + 16*j，j=0..9
    channel_max: int = 8
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    train_ratio: float = 0.6                 # 真实数据时间切分
    diag_ratio: float = 0.2
    cert_ratio: float = 0.1
    diag_a_ratio: float = 0.5                # diag 内 A 段占比
    seed: int = 0

    @property
    def max_lag(self) -> int                 # lag_segments 最大值（335）
    @property
    def burn_in(self) -> int                 # max_lag + fir_len_cap
    def validate(self) -> list[str]          # 汇总全部错误，空表为合法
    def to_dict(self) -> dict[str, Any]      # camelCase JSON
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HeadroomConfig"   # 未知键报错
    @classmethod
    def from_json(cls, path: Path) -> "HeadroomConfig"            # 解析失败/校验失败 ValueError
```

`validate` 只保留违反后会**静默污染结果**的规则（不做逐字段区间防御）：

- `burn_in >= input_length`：独立 cert/test 段的输入窗口从段首起算，burn-in 必须盖住整个 lookback；
- `burn_in >= input_length + 16*(beyond_l_lags - 1)`：beyond-L 特征的最深滞后不得越出段内已实现范围（否则 numpy 负下标会静默绕到未来行）；
- `burn_in >= horizon + horizon_realized_lags - 1`：horizon 特征同理；
- `band_period_range[1] // 4 <= max_lag`：band 坐标读取的 P//4 相位滞后必须落在 lag 覆盖内。

## 3. `headroom/timeidx.py`

```python
@dataclass(frozen=True)
class Splits:
    train: slice
    diag: slice
    cert: slice
    test: slice

def chrono_splits(values_len: int, cfg: HeadroomConfig, embargo: int) -> Splits
    # 真实数据：train/diag/cert/test 按比例依时间排列，相邻段间插入 embargo 步
    # （embargo 属于后段的开头）。test 为最后一段。

def segment_origins(seg_start: int, seg_stop: int, cfg: HeadroomConfig) -> np.ndarray
    # 可用起点 o：seg_start + burn_in <= o <= seg_stop - H - 1（目标完全落在段内）

def windows(values: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig)
    -> tuple[np.ndarray, np.ndarray]
    # (X[B,L,C], Y[B,H,C])；输入窗口允许延伸到段之前的数组前缀（已实现的过去）

def ab_split(origins: np.ndarray, a_ratio: float, embargo: int)
    -> tuple[np.ndarray, np.ndarray]
    # 按时间把起点切成 A、B 两段，中间空开 embargo 个起点
```

## 4. `headroom/residuals.py`

```python
@dataclass(frozen=True)
class SegmentData:
    values: np.ndarray     # [T, C] 该段完整数组（含前缀）
    origins: np.ndarray    # [B] 可用起点（局部下标），升序
    r: np.ndarray          # [B, H, C] 与 origins 对齐的残差
    e_full: np.ndarray     # [T, C]；仅在「目标完全落在段内」的起点处非 NaN
    r_full: np.ndarray     # [T, H, C]；同上，NaN 填充

def segment_data(values: np.ndarray, seg_start: int, seg_stop: int,
                 pred: np.ndarray, cfg: HeadroomConfig) -> SegmentData
    # pred[B,H,C] 与「目标完全落在段内」的全部起点 [seg_start, seg_stop-H-1] 对齐
    # （注意：不是 burn_in 限制后的可用起点——e/r 序列必须从段起点就存在，
    # 否则首个可用起点的深滞后特征会缺失）。origins 字段才是 segment_origins 的输出；
    # 两者行数不同，长度不符抛 ValueError。
```

**截断不变性**（必须测试）：任意坐标在起点 o 的特征只依赖 `values[0..o]`、`e_full[0..o]`、`r_full[0..o]`。

## 5. `headroom/correctors.py`

```python
@dataclass(frozen=True)
class Coordinate:
    kind: str    # total | lag | band | beyond_l | channel | horizon | hgroup
    label: str   # 例如 "lag[0,23]"、"band(P=24)"、"beyond_l"、"channel->3"、"horizon"、"hgroup[1,12]"
    aspect: str   # periodicity | lag-structure | long-range-beyond-window | cross-channel | horizon-range | level-bias | other
    spec: dict    # 如 {"k1": 0, "k2": 23} / {"period": 24} / {"target_channel": 3} / {"h1": 1, "h2": 12}

def select_period_peaks(e_full: np.ndarray, a_origins: np.ndarray, cfg: HeadroomConfig) -> list[int]
    # 只用 A 段（a_origins 为 A 段起点下标，连续）：平均各通道周期图，取 band_period_range
    # 内按高度排序的局部极大峰；接受一个新峰要求它与已接受峰的**周期相对间隔 >= 15%**
    # （同一谱瓣只报一次，长周期与短周期一视同仁）；相邻 bin 舍入到同一整数周期的重复
    # 剔除；返回至多 max_period_peaks 个周期（整数步）

def enumerate_coordinates(cfg, period_peaks: list[int], n_channels: int,
                          horizon: int) -> list[Coordinate]
    # = lag 段坐标 + band(P) 坐标 + beyond_l + 每个通道一个 channel(->i) + horizon
    # n_channels < 2 时不生成 channel 坐标（没有其他通道可喂，会退化为纯截距拟合）
    # （total/hgroup 由认证族与 corrector_gain 单独处理，不在本表）

def build_features(seg: SegmentData, coord: Coordinate, cfg: HeadroomConfig)
    -> tuple[np.ndarray, np.ndarray]
    # (X[B,F], Y[B,H,C])，特征只来自该坐标：
    # lag[k1,k2]   : e_full[o-k]（k 取 [k1,k2] 内，第 3 段步长为 2），目标全 (h,c)
    # band(P)      : e_full 经因果 FIR 带通（firwin，numtaps=min(fir_len_cap,4P) 取奇数，
    #                通带为角频率 [2π/(1.2P), 2π/(0.8P)]；firwin 的 cutoff 以 Nyquist 为单位
    #                （1.0 = π rad/样本），故实际传 [2/(1.2P), 2/(0.8P)]），
    #                取滞后 0 与 P//4 两个相位
    # beyond_l     : values[o-k] - values[o]，k = L + 16*j（j=0..beyond_l_lags-1）
    # channel->i   : 其他通道的 e_full[o-k]（k=0..23）与 values[o-k]-values[o]（k=0..23），
    #                目标仅通道 i（其余通道的目标列置零：零目标列对任意 alpha 得零 ridge
    #                列，不影响 alpha 选择，也不影响通道 i 的拟合）
    # horizon      : 对每个 h，特征 r_full[o-h-k, h]（k=0..horizon_realized_lags-1）；
    #                实现为逐 h 的独立 ridge（见下）
    # kind 为 total/hgroup 时抛 ValueError：聚合统计量走 fit_greedy/corrector_gain

@dataclass(frozen=True)
class GainReport:
    coordinate: Coordinate
    delta: float            # eval 段实现增益（正=改进）
    lcb: float              # HAC+t 解析置信下界（在下方 level 处计算）
    ucb: float
    level: float            # lcb/ucb 的置信水平；certify 逐项校验它等于 certify_level
    by_horizon: np.ndarray  # [H] 逐 h 的增益
    mu2: float               # eval 段 (mean r)²
    n_eval: int

def coordinate_gain(coord: Coordinate, seg_a: SegmentData, seg_b: SegmentData,
                    cfg: HeadroomConfig, level: float = 0.95) -> GainReport
    # 单坐标增益。A 段：特征内部再按时间 8:2 切分（间隔 H）选 ridge alpha，然后全 A 重拟合；
    # B 段：评估逐起点损失差 d_o = mean_{h,c}(r² - (r-ĝ)²)，delta=mean(d)，
    # lcb/ucb = inference.mean_ci(d, bandwidth, level)，带宽下限 = max_lag + H
    # （增益序列继承残差过程的记忆：季节任务的 d_o 自相关可延伸到周期长度）。
    # horizon 坐标：逐 h 拟合/评估，delta = mean over h。channel 坐标：d 只对通道 i 求平均。
    # total/hgroup 不是单坐标，不走本函数（build_features 已拒绝）。

def _fit_ridge(x: np.ndarray, y: np.ndarray, alphas: tuple[float, ...],
               embargo: int) -> tuple[np.ndarray, np.ndarray]
    # 带截距 ridge（私有；coordinate_gain 与 fit_greedy 的最终重拟合共用）：
    # 列按拟合部分均值中心化；alpha 在时间序 80/20 切分（fit 与 eval 行间隔 embargo，
    # 调用方传 H）上按验证 MSE 选，胜出 alpha 用全部行重拟合。
    # 均值/标准差标准化不在本函数——fit_greedy 的贪心块已提前做好。

@dataclass(frozen=True)
class FittedCorrector:
    coordinates: tuple[Coordinate, ...]  # 贪心选中的坐标（按选择顺序）
    means: np.ndarray     # [F] A 内拟合行的特征均值（标准化参数）
    stds: np.ndarray      # [F] A 内拟合行的特征标准差（0 记 1）
    coef: np.ndarray      # [F, H*C] 标准化空间中的 ridge 系数
    intercept: np.ndarray # [H*C]
    label: str = "total"

def fit_greedy(seg_a: SegmentData, cfg: HeadroomConfig, period_peaks: list[int],
               *, max_steps: int = 4) -> FittedCorrector
    # total = 坐标的**贪心前向选择**，不再是全部特征的并集（并集 ridge 在低信号任务上
    # 过参数化、样本外负迁移，已废弃）。候选池 = enumerate_coordinates 去掉 horizon；
    # 每个坐标的特征块按 A 内拟合行（前 80%）的均值/标准差标准化，只预计算一次；
    # 在 A 内 80/20 时间切分（间隔 H embargo）上以验证 MSE 逐个加入最能改进的坐标
    # （alpha 在 cfg.ridge_alphas 中取最优），至多 max_steps 步、无改进即停
    # （空选择 = 仅截距基线）；选中的坐标集最后用 _fit_ridge 在全部 A 行上重拟合，
    # 标准化参数存入 FittedCorrector。选择只看 A，eval 段（B/cert/test）全程干净。

def corrector_gain(fitted: FittedCorrector, seg_b: SegmentData, cfg: HeadroomConfig,
                   level: float = 0.95, *, h_scope: tuple[int, int] | None = None,
                   label: str | None = None) -> GainReport
    # 把拟合好的校正器应用到 eval 段（各坐标特征按 means/stds 标准化后过 coef+intercept），
    # d_o 同上。h_scope=(h1,h2) 时 d 只对 h∈[h1,h2] 求平均，Coordinate 记 kind="hgroup"
    # （hgroup 统计量 = 同一 greedy 校正器、限定步长区间的 scope）。

def all_gains(seg_a, seg_b, cfg, period_peaks, level: float = 0.95) -> list[GainReport]
    # greedy total + 全部坐标 + hgroup[1, H//2] + hgroup[H//2+1, H]；
    # total 与两个 hgroup 共享同一个在 A 上拟合的 greedy 校正器，hgroup 只把评估
    # 限制到各自步长区间（与认证语义一致）

def certification_family(cfg, period_peaks: list[int], n_channels: int) -> list[Coordinate]
    # 认证统计量族 K：total + enumerate_coordinates 的全部坐标 + 两个 hgroup；
    # total/hgroup 的 spec["periods"] 携带 A 上选出的周期峰

def certify_level(cfg, j_max: int, period_peaks: list[int], n_channels: int) -> float
    # 认证增益的逐统计量 Bonferroni 水平 1 - (certify_alpha / j_max) / |K|（见 §7）

def card_level(cfg, period_peaks: list[int], n_channels: int) -> float
    # 卡片坐标增益的 Bonferroni 水平 1 - acceptance_alpha / |坐标数|（见 §9）
```

## 6. `headroom/inference.py`

```python
def hac_variance(d: np.ndarray, bandwidth: int) -> float
    # Newey–West（Bartlett），对均值的标准误

def auto_bandwidth(d: np.ndarray, min_bw: int) -> int
    # 简单规则：max(min_bw,Politis–White 类估计)；实现者可取 floor(4(n/100)^(2/9)) 类公式

def mean_ci(d: np.ndarray, bandwidth: int, level: float) -> tuple[float, float, float]
    # (mean, lcb, ucb)：HAC 标准误 + t(n-1) 临界值，带 HLN 小样本因子
    # sqrt((n + 1 - 2K + K(K-1)/n) / n)，K=bandwidth（注意外层的 /n；
    # n >> K 时略小于 1，使区间略变宽、检验保守）

def dm_one_sided_p(d: np.ndarray, bandwidth: int) -> float
    # H0: mean(d) <= 0（无改进）；HLN 修正后的单侧 p 值
```

## 7. `headroom/certify.py`

> 2026-10-16 §13 修订：删除 α 消耗记帐（跨版本均匀分配 + Bonferroni 摊销 + 累计误停界）；
> 停止规则退化为固定水平检验，误停率为 G0-1/G0-5 的实测性质。

```python
@dataclass(frozen=True)
class CertReport:
    saturated: bool
    level: float            # 固定水平 1 - certify_alpha
    failures: list[dict]    # [{"label", "ucb", "threshold"}] 未通过项

def certify(gains: list[GainReport], mse_by_group: dict[str, float],
            cfg: HeadroomConfig) -> CertReport
    # 一句话规则：cert 段上每个统计量的可实现增益 UCB 都 < certify_epsilon * 该项 MSE
    # 即判定饱和。固定水平，无跨版本/跨统计量预算。
    # gains 必须已按 level=1 - certify_alpha 预先计算（all_gains）；certify 逐项校验
    # gain.level 一致，不一致抛 RuntimeError（水平不对，规则就变了）。
    # mean_ci 是双侧区间，因此每个 UCB 实际是单侧 1 - alpha/2 置信界：只在让停止
    # 更保守的方向上偏。
    # mse_by_group：cert 段上按每项 scope 的平均残差 MSE——total→全部 (o,h,c) cell，
    # hgroup[h1,h2]→其步长区间，channel->i→通道 i，其余坐标→全部 cell
    # （experiments/evaluate.py::mse_by_group 实现该 scope）
```

## 8. `headroom/synthetic.py`

```python
class Oracle(Protocol):
    def forecast(self, seg: np.ndarray, origins: np.ndarray,
                 cfg: HeadroomConfig) -> np.ndarray   # [B, H, C] 全信息最优预测
    def mse_theoretical(self) -> np.ndarray | None     # [H] 或 None

@dataclass(frozen=True)
class SyntheticTask:
    name: str
    train_diag: np.ndarray   # [T, C] 连续 train+diag；train=[0,diag_start)，diag=[diag_start,T)
    diag_start: int
    cert: np.ndarray         # 独立新模拟，前缀 >= burn_in + L
    test: np.ndarray         # 独立新模拟
    oracle: Oracle
    structure: dict          # {"periods": [...], "cross_lags": {...}, ...}，供映射表使用

def lgssm_multiperiod(seed, sizes) -> SyntheticTask
    # C=4。潜变量：AR(1) 水平(φ≈0.995) + 周期 24 与 168 的正弦对（转角伴式）；
    # 观测 = 各通道对潜变量的不同线性组合 + 通道 AR(1) 噪声 + 观测噪声；
    # 通道 3 不含水平分量（部分观测）。参数已知 → Kalman 从段起点滤波 + h 步外推。
def lgssm_xchannel(seed, sizes) -> SyntheticTask
    # C=4；y_t = Σ_j A_ij y_{j,t-lag_ij} + 通道噪声，lag_ij ∈ {3,5,7} 各对不同；
    # 伴式 VAR(max lag) Kalman（参数已知）。
def hmm_ar_switch(seed, sizes) -> SyntheticTask
    # C=1，K=2 体制 AR(1)（φ∈{0.3,0.95}，均值 {0,2}），粘性转移阵，观测无噪声；
    # oracle 用联合递推 m_s(o)=E[x_o·1{S_o=s}]（O(K²) 每步），不得用「按后验混合预测」近似。
def arfima(seed, sizes) -> SyntheticTask
    # C=1，d=0.35；分数差分模拟；oracle = 理论自协方差上 Durbin–Levinson 到阶 P>=2000
    # 得 AR(P) 系数，h 步用伴式矩阵幂。信息深度 P >= burn_in。
def garch(seed, sizes) -> SyntheticTask      # C=1；oracle ≡ 0
def random_walk(seed, sizes) -> SyntheticTask  # C=1；oracle = 最后观测值

REGISTRY: dict[str, Callable[..., SyntheticTask]]
# sizes = {"train": 20000, "diag": 10000, "cert": 12000, "test": 20000}（默认）；
# 测试可用小 sizes。所有生成器同 seed 严格可复现。
```

oracle 正确性检验（单测）：LGSSM 的 Kalman h 步预测与高蒙特卡洛/解析一致；HMM 联合递推与小规模暴力枚举一致；ARFIMA 的 AR(2000) 一步预测与真实新息近似正交；m=oracle 时 gap≈0（stage0）。

## 9. `headroom/feedback.py`

> 2026-10-16 §12 修订：bj 臂退役；card 更名 fixed；新增 self-built 臂（agent 自建探针的
> 输出作为其中段内容，探针契约见 experiments/prompts/probes.md 与 experiments/probes.py，
> 本节其余文字以修订前的四臂设计为历史记录保留）。外框一致性规则不变。

```python
def render_feedback(arm: str, incumbent: dict, gains: list[GainReport],
                    history: list[dict], cfg: HeadroomConfig,
                    placebo_perm: tuple[int, ...] | None = None,
                    probe_rows: list[dict] | None = None,
                    validation_rows: list[dict] | None = None) -> str
    # 返回 markdown。外框四臂一致：标题、incumbent diag MSE、按 hgroup 的 MSE、
    # 历史候选表（round, id, accepted, diagMse）。
    # scalar：无附加内容。
    # self-built：上轮探针校验结论（kept/rejected）+ 已入库探针在 incumbent 上的最新输出
    #     （最新 8 行、每行 ≤600 字符）；后端诊断内容不进入该臂。
    # bj（已退役）：1 步残差 ACF 前 48 阶（按 lag 顺序）、逐通道 Ljung–Box（按 lag 渲染 Q 与跨通道
    #     最小 p）、周期图前 5 峰；不含任何增益/排序。
    # card：坐标增益行 = 点估计 + 解析置信下界，按调用方传入的 LCB 降序排列——调用方在
    #        correctors.card_level(cfg, period_peaks, n_channels) 的 Bonferroni 水平上计算
    #        卡片增益，本函数只排序、不重算区间；另附 total 的 delta 与 LCB、μ²、
    #        各坐标 by_horizon 的峰值区间；使用 Coordinate.aspect 措辞。
    #        hgroup 增益不进卡片（保留给认证）。
    # placebo：与 card 完全同构，但用 placebo_perm 固定置换坐标行（内容取自同一张卡片）。
    # incumbent dict：{"diagMse": float, "mseByHGroup": {...}, "round": int, "budgetLeft": int,
    #                 "eFull": np.ndarray}（eFull = diag 段 e_full，bj 臂据此自算 ACF/Ljung–Box/周期图）
    # history row：{"round", "candidateId", "accepted", "diagMse"}
    # 禁止出现：数据集名称、领域词（ECG/电力等）、「残差」之外的引导词表按计划 §3.2。
```

## 10. `headroom/usage.py`

```python
def session_usage(path: Path) -> dict
    # 读单个 session.jsonl.zstd：累加每条 assistant 事件 data.usage 的
    # {inputTokens, outputTokens, cacheReadTokens, reasoningTokens}，并计 steps。
    # 文件不存在 → FileNotFoundError；zstd 不可用 → RuntimeError（消息含安装指引）。

def workspace_usage(workspace: Path) -> dict
    # 递归汇总 workspace/.dsh-sessions/**/session.jsonl.zstd
```

## 11. 风格

- `from __future__ import annotations`；dataclass + 完整类型标注；无 print；模块级 docstring 一句话。
- 参照 `backend/harnessing_ts/node_state.py` 的语气写注释：解释「为什么」，不复述代码。
- 单测放在 `tests/test_headroom_<module>.py`，只用 numpy/scipy/pytest（usage 用 zstandard 造 fixture）。
- 数值断言给出容差并注释来源；涉及模拟的测试固定 seed 且控制规模（单文件 < 20s）。
