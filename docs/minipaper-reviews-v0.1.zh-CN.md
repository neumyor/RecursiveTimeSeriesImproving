# Mini-paper v0.1 三方评审汇总

> 对象：[minipaper-residual-guided-discovery.zh-CN.md](minipaper-residual-guided-discovery.zh-CN.md)（v0.1）
> 评审：R1 时序统计专家 / R2 LLM agent 与程序演化专家 / R3 资深 AC
> 日期：2026-10-15。评审中引用的文献来自评审者的网络检索，尚未逐篇核对原文。

## 1. 打分

| | Novelty | 技术性 | 时序相关性 / 意义 | Clarity | Overall | Confidence |
|---|---|---|---|---|---|---|
| R1 | 5 | Soundness 4 | TS 相关性 7 | 7 | **4** | 4 |
| R2 | 5 | Tech. Novelty 4；实验设计 4 | — | 7 | **4** | 4 |
| R3 | 5 | Depth 4（补命题后 6） | Significance 6 | 7 | **5** | 4 |

结论：当前为 weak reject / borderline。三位都认为核心洞见成立、可一眼看懂，问题集中在**过度声明**、**统计细节**和**基线与归因**。

## 2. 三位一致的问题（必须改）

### C1. P1 不是时序独有，“对比”段有事实错误
- “残差能被审计器预测 ⇒ 非最优”在一切监督学习中成立：multicalibration / outcome indistinguishability（Hébert-Johnson 2018；Globus-Harris ICML 2023）、boosting、RL 的 Bellman 残差、空间统计的 Moran's I、Brier 分数的 Murphy 分解。
- “图像分类最优分类器错误无统计结构”“CV/NLP 无法构造最优已知任务”都不成立（高斯混合、PCFG/HMM 语言的熵率）。
- **共同建议的改述**：时序是三件事**同时、零成本、与模型无关**成立的领域：
  1. 残差自身的过去就在信息集里，是免费、通用的审计输入（iid 任务必须人工挑审计器类）；
  2. 线性部分有闭式剩余空间（KS / 谱平坦度，多变量用 Whittle 的 log-det 形式）；
  3. 频率 / 滞后 / 步长 / 通道是跨数据集通用、并能对应到结构机制的坐标。
- 把 multicalibration 等写进相关工作，作为“同一原理在其他领域的形式”。R1 建议加一个“通用审计器反馈”消融，用实验说明收益来自时序坐标。

### C2. “最优性证书”“已认证饱和”言过其实
- $H>0$ 只能证明**非最优**；$H\approx0$ 只说明“在探针类 $\mathcal{G}$ 和信息集下没测出来”。不拒绝原假设 ≠ 接受原假设，diag 越短越容易“认证饱和”。
- 存在逐轮重复检验（optional stopping）、多频段 / 多步长的多重比较、diag 反复使用的自适应过拟合问题。
- **建议**：改成“非最优性见证 + $\mathcal{G}$-相对的 ε-饱和”；停止条件改为 headroom **上置信界 < ε·MSE**（等价检验 / anytime-valid / α-spending）；写成命题：$P(\text{停止时 } H_\mathcal{G}>\varepsilon)\le\alpha$。

### C3. 频段分解“精确”但不可归因
- $\sigma^2_{\mathrm{KS}}$ 是全局几何均值，修好一个频段会让所有 $H_b$ 都变；负贡献频段直接丢掉，比例的分母也不清楚。
- 修改方案（二选一）：
  - R1 / R3：用逐点非负的 **Itakura–Saito 形式**：$H_{\mathrm{lin}}-\mu^2=\frac{1}{2\pi}\int \sigma^2_{\mathrm{KS}}(x-1-\log x)\,d\omega$，$x=f/\sigma^2_{\mathrm{KS}}$；
  - R2：改用**可实现的分坐标校正器增益**（只用频段 b 带通后的过去残差拟合线性校正器，取它带来的 MSE 降幅）。
- 卡片里的“理论 1.3×”在真实数据上算不出来，要删掉。

### C4. 缺少会直接威胁 LLM 作用的基线
| 基线 | 提出者 | 用来检验什么 |
|---|---|---|
| $m+g$ 迭代残差 boosting（不用 LLM） | R1 R2 R3 | headroom 本来可以直接兑现，LLM 改结构多带来了什么 |
| 原始 Box–Jenkins 诊断文本（ACF / 周期图 / Ljung–Box，不给 headroom） | R2 R3 | 证书语义和排序是否有贡献，避免被说成“prompt 加 Ljung–Box” |
| 规则修复（缺陷 → 固定修法，不用 LLM） | R3 | P2 的映射表若可确定性执行，LLM 的作用在哪 |
| auto.arima / ETS / Theta / 状态空间 | R1 R3 | 合成高斯线性任务上的强经典基线 |
| 零样本 TSFM（Chronos-2 / TimesFM / Moirai） | R3 | “为什么不直接用基础模型” |
| 安慰剂卡片、可自由写分析代码的 agent | R2 | LLM 是否真的在用卡片的内容 |
| EvoCast / SEATS（含其 MAP-Elites） | R2 | 最接近的时序结构演化工作 |

还有一个 Goodhart 漏洞：agent 把探针 $g$ 原样接成校正头，按构造就能通过机制一致检验，必须识别或报告。

### C5. 合成任务的“最优已知”不严谨
- Lorenz 加观测噪声时，最优误差大于噪声方差；“已知状态的漂移正弦”用了预测者看不到的信息；有限 lookback 下 ARMA 的最优误差大于新息方差；高阶 AR 太容易，各方法曲线会挤在一起。
- **建议**：主力改用**线性高斯状态空间（Kalman 精确最优）**与 **HMM 体制切换（前向滤波精确最优）**、ARFIMA；最优参照用“L-受限最优 + 正确设定模型在同一训练集上的拟合误差”。另加两类：“白但非线性可预测”的过程（双线性、TAR），以及负对照（GARCH；Exchange-Rate 应在第 0 轮就判定饱和）。

### C6. 头条“打破固着”选错了
- 和“经典修法也可接受”的立场有张力；“设计空间外”靠作者自己标注，主观；因果链长，缺少证据。
- **建议**：头条改成“**数据定向、可认证停止的预测模型发现**”。固着、off-target 率、修改类型降为分析结论。如果保留“空间外”，就分成 out-of-recipe 和 out-of-literature 两级，并由作者之外的人盲标。

### C7. 定位步骤（3.2）脆弱，建议降级
- 线性 embedding 满秩时，第 1 层的线性 R² 不会比输入低，所以“第 1 层骤降”几乎不会出现；
- 短窗口带通会带来边界泄漏；表示维度高而样本少，需要 control task / MDL 探针；
- DLinear 没有层，LLM 生成的分支结构没有层序。
- **建议**：主方法只保留“诊断 → 修复 → 认证停止”；定位作为分析工具和消融，并先在植入缺陷的模型上验证定位准确率。

### C8. 统计协议
- 重叠窗口的配对检验要用 Diebold–Mariano + HAC（Newey–West，滞后 ≥ h−1）或 block bootstrap；
- 每组至少 5–10 次独立运行；
- 横轴同时给出候选数、GPU 时长和 token；
- 报告 diag 与 test 增益的散点；
- 真实数据以 GIFT-Eval / fev-bench 为主，ETT / Weather 有已知问题。

## 3. 各评审独有的关键意见

**R1（时序统计）**
- 在有限样本下，$H_{\mathrm{lin}}$ 其实是**可实现修正量的上界**（要扣掉 $O(p/n)$ 的估计误差），它只是“离 Bayes 最优差距”在总体意义上的下界。需要 L-受限的版本（Durbin–Levinson / AR-sieve）。
- “线性情形下是白噪声”的表述方向反了：条件期望残差是 MDS，一定是白的；白化是必要条件，不是充分条件。
- 估计细节：剔除 ω=0 和 π；强谱峰要加锥窗或 multitaper（修正项变成 $\log K-\psi(K)$）；$\log\hat\sigma^2_{\mathrm{KS}}$ 的标准误约 $1.81/\sqrt n$，在 ETTh1 上和 headroom 同量级。
- 多步：信息集应包含**所有已实现的跨步长残差** $\{r^{(j)}_{t-k}:k\ge j\}$。另外“频率”混用了两种含义（沿起点的残差谱 vs 沿步长的误差轨迹谱），要分开定义。KS 只覆盖 h=1，和 LTSF 的 96–720 步主指标关系不大。
- 补充相关工作：ForeCA、加权排列熵可预测性、**Spectral Coherence Predictability（2509.23074，和 $H_b$ 非常接近）**、Automatic Statistician（诊断驱动结构搜索的先驱）、Sun et al. NeurIPS 2021、FreDF。

**R2（LLM agent / 演化）**
- **最接近的已有工作**（必须正面讨论）：
  - **Deliberate Evolution**（ICML 2026, 2606.04360）：符号回归里用残差诊断工具定向修改，动机几乎相同，但没有最优性语义；
  - **EvoCast**（2610.04517，2026-10-03）：时序**结构演化** agent，诊断用的是机制消融和数据特征，带可检验假设和确定性晋升规则，**和本文方法论最接近，被抢先的风险最高**；
  - **CTRL**（ACL Findings 2026）：把残差分解统计交给 LLM，但修正的是输出，不改结构；
  - **MARICL**（2605.22897）：表格数据上从残差生成机制假设。
- R2 的结论：完全相同的组合还没有，但能守住的新颖性只剩两点：①最优性语义（headroom + 坐标分解 + 认证停止）；②机制一致接受。
- **归因**：RESIDUAL 和 Scalar 之间不止一个变量不同（反馈、接受规则、写预期、定位信息、算力）。最小实验应做成**析因 4 组**：Scalar / Scalar + 原始诊断 / Card + MSE 接受 / Card + 机制接受，并要求所有组都写预期。
- H4 在构造上必然成立，是循环论证。改为在**所有组的全部提案**上事后计算“缺陷定向率”。
- 一个极简的定向性模型：命中真实缺陷类的期望候选数，从 $1/p(k^\star)$ 降到 $1/p(k^\star\mid\text{card})$。这同时给出了固着的可测定义。
- 把缺陷卡作为插件接入 OpenEvolve / ShinkaEvolve，证明它和搜索策略正交。
- 真实数据集在预训练语料里，LLM 本来就知道 ETTh1 有日周期，需要**去标识化**，或者构造“流行做法有害”的先验对抗型合成任务。

**R3（AC）**
- **建议补的命题**：
  - 命题 1：headroom 夹逼 $0\le H_\mathcal{G}\le \mathrm{MSE}-\mathrm{MSE}^\star$，线性类时有闭式，附有限样本置信界；
  - 命题 2：ε-饱和停止的误停概率不超过 α；
  - 观察 0：可以构造 MSE 相同、但一个已经最优、一个还差 X% 的两个模型，标量反馈无法区分，残差可以。
- **最重要的一张图**：在最优已知的合成任务上，横轴为候选数，纵轴为真实最优性差距；同时叠加估计的 H 和置信区间，并标出停止点。这张图同时证明证书准、反馈有用、停止正确。
- **Venue**：首选 KDD 2027 Research Track；ICML 2027 需要命题写扎实；最优已知合成平台可以单独投 NeurIPS D&B。
- **标题建议**：*Is Your Forecaster Done? Residual Headroom as a Free Certificate for LLM-Driven Forecasting Model Discovery*。

## 4. 修改优先级（v0.2）

1. **重写叙事和理论语义**：头条改为“缺什么、在哪、何时停”；P1 承认普遍性，改为“三性质零成本共存”；证书改为 $\mathcal{G}$-相对 ε-饱和；补命题 1、2 和观察 0；频段分解换成 IS 形式或可实现增益。
2. **方法做减法**：只保留诊断 → 修复 → 认证停止；定位降为分析；领域坐标默认关闭；明确 L-受限信息集和跨步长残差。
3. **重做最小验证实验**：Kalman / HMM 精确最优合成族；析因 4 组 + $m+g$ boosting + 原始诊断文本；主图为“估计 H vs 真实差距 + 停止点”。先确认 $m+g$ 不能追平 LLM 改结构。
4. **相关工作**：先细读 EvoCast 与 Deliberate Evolution，确认差异点；补上 multicalibration、Automatic Statistician、Spectral Coherence Predictability、CTRL、CRAFTER。
