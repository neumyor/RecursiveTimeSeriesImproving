# Is Your Forecaster Done? 用残差 headroom 告诉 LLM agent 缺什么、在哪、何时停

> Mini-paper 草稿 v0.4（2026-10-16）。目标：KDD 2027 Research Track（备选 ICML 2027）。
> v0.1 → v0.2 的修改依据见 [三方评审汇总](minipaper-reviews-v0.1.zh-CN.md)；v0.2 → v0.3 的修改依据见 [最终编码计划](residual-headroom-final-coding-plan.zh-CN.md)。
> **v0.3 → v0.4 的动因是 2026-10-16 的方向修订（作者指定，见编码计划 §12）与 M4 三方代码评审**；变更如下。

**v0.4 变更（相对 v0.3）：**

1. **主循环从“两处不同”改为三个共享机制**（摘要 / §1 / §3.4）：反馈通道（缺陷卡或 agent 自建探针）、认证停止、残差签名演化引导。三臂共用后两个机制，臂间差异只在反馈通道——这既是诚实描述，也是 M4 评审 B 指出的必要修正。
2. **证书可见性主张修正**（§3.3 / §3.5）：v0.3 声称“cert 不以任何形式返回给 agent、不需要为复用一个比特建模”。方向修订把证书饱和作为演化引导的触发器之一后，**一个比特（饱和与否）确实进入所有臂的提示**。数值仍不可见；停止点仍由后台判定；但 RQ3 的“停后继续搜索的收益”从此在**主动多样性协议**下测得，论文按此如实表述。
3. **对比臂更新**（§4.3）：Scalar / Fixed（原 Card，后端固定卡）/ **Self-built（新主线：agent 自建、植入信号校验的诊断探针）**；Scalar+BJ 退役。核心对比改为 **self-built vs fixed**——"agent 自建并经校验的工具能否胜过后端固定工具”。诚实披露：该对比捆绑了两个差异（谁建工具 + 自建工具可读逐层激活），不拆分激活通道。
4. **T1 论证补强**（§2）：Deliberate Evolution 对“审计器类必须人为挑选”的批评同样适用于自建探针臂——agent 也在挑自己的审计类。我们的回答是：时序里免费且通用的是**审计输入**（残差过去 + 隐层激活）与**校验协议**（植入信号夹具上的方向性、裕度与特异性检验），探针只要能通过校验就承认，不预设工具形态。
5. **初始 incumbent 换为 PatchTST-lite**（§3.4 / §7）：隐层分析从消融升为主线（自建探针可读 256 个对齐窗口的逐层激活），相应地 G0 中依赖 m0 的三个门（紧度 / 定位 / 筛选）用深度缺陷变体重校准。
6. **新增机制描述**：探针契约与校验（§3.6，附录 D 全文引用探针指南）；残差签名档案与多样性切换（§3.7）；新增诊断量（切换次数、切换后首接受是否落在不相交坐标族、探针账本；§4.4）。
7. **RQ2/RQ3 措辞对齐**（§4.1）；scalar 臂的描述修正为“标量反馈 + 共享引导”（其脚手架含按步长分组的 MSE 表）；M4 评审要求的披露（激活通道捆绑、自建臂首轮自举延迟、探针渲染上限）写入 §4.3。
8. **删除认证停止的 α 记帐**（修订 II，作者指定）：跨版本均匀分配 + Bonferroni 摊销 + 命题 2 的累计误停界全部移除；停止规则退化为固定水平检验（§3.5 重写），误停行为改为 G0-1/G0-5 的实测性质，不再主张解析保证。

---

## 摘要

LLM agent 已经能自主编写和迭代时序预测模型，但驱动它的几乎总是一个验证集分数。分数只回答“差多少”，回答不了三个问题：还缺什么、缺在哪里、是否已经没什么可改。

我们指出，时序预测的残差可以免费回答这三个问题。对任意预测模型 $m$，凡是能用“预测时刻已知的信息”去预测 $m$ 的残差的校正器，都直接构成一个更好的预测器。因此，**残差可被预测的程度（residual headroom）是 $m$ 距离最优的一个可实现下界**。

这一原理本身适用于所有监督学习，但时序预测是它**零成本、与模型无关、且坐标规范**的领域：
- 残差自身的过去天然属于信息集，是一个现成的通用审计输入；
- 线性部分有闭式解（Kolmogorov–Szegő）；
- headroom 可以沿频率、滞后、预测步长、通道分解，而这些坐标能直接对应到结构机制。

基于此，我们提出一个极简的 agent 循环。它与普通“标量反馈”循环在**三个机制**上不同，其中后两个为所有对比臂共享：
1. **反馈通道**：按坐标分解、带置信下界的**缺陷卡**（后端固定工具），或 **agent 自建的、在植入信号夹具上通过方向性与特异性校验的诊断探针**——工具本身成为搜索对象的一部分；
2. **认证停止**：当每个残差坐标的剩余 headroom 上置信界都低于阈值的固定比例时停止——一条规则、一个固定水平；
3. **残差签名演化引导**：把每个已评估候选按“它关闭了哪些残差坐标”编入档案；当证书饱和、或连续接受集中在同一坐标族时，改推**签名不相交**的最佳历史候选作为起点——多样性轴就是证书测量的时序坐标空间本身。

我们构建了一组**最优预测器可以精确计算**的合成任务（线性高斯状态空间、隐马尔可夫体制切换等），在上面直接测量四件事：估计的 headroom 是否跟踪真实差距、哪种反馈更快逼近最优、停止时是否真的接近最优、自建工具能否胜过固定工具。随后在去标识化的真实任务上验证。

---

## 1. 引言

**现象。** 现在的时序 agent 有的选择模型（TimeSeriesScientist、TimeCopilot），有的演化代码或结构（SEATS、EvoCast），但都以标量指标作为核心反馈，以预算耗尽作为停止条件。

**问题。** 标量反馈缺三样东西：
- **方向**：分数不说明缺的是周期、长滞后还是跨通道依赖。agent 只能依赖文献先验，而时序文献的实证结论本身噪声很大（2512.22702 指出，实现细节的影响常常超过换一种序列层）。
- **定位**：分数不区分哪个预测步长、哪个通道、哪个频段出了问题。
- **终点**：agent 不知道还剩多少可改。它可能在噪声底上空转，也可能在还有大量空间时就过早放弃。

**观察 0（标量不可辨识）。** 取两组“过程 + 模型”：
- (A) $x_t$ 是方差为 $s^2$ 的白噪声，模型 $m_A\equiv 0$；
- (B) $x_t=\phi x_{t-1}+e_t$，$\mathrm{Var}(e_t)=s^2(1-\phi^2)$，模型 $m_B\equiv0$。

两者 MSE 都是 $s^2$，标量反馈无法区分。但 $m_A$ 已经是最优，$m_B$ 离最优还差 $\phi^2 s^2$。这个差距恰好等于 $m_B$ 残差的一步线性可预测量，残差一眼就能看出来。

**思路。** 这正是 Box–Jenkins “识别 → 估计 → 残差诊断 → 再识别”循环的核心。我们把它变成 LLM agent 的反馈和停止信号，并补上它原本没有的三样东西：按坐标分解、可逐项兑现的 headroom；有统计保证的停止规则；以及**让 agent 自己构建诊断工具**的通道——工具必须先在植入已知信号的合成夹具上通过校验（能检出、有裕度、且对无信号对照不误报），才有资格成为反馈。

**贡献。**
1. **原理**：定义相对于信息集和校正器类的残差 headroom，证明它是到最优距离的可实现下界（命题 1）；给出按坐标分解、可逐项兑现的版本，以及 ε-饱和停止规则；论证时序是审计输入、闭式解与结构坐标三者同时零成本成立的领域（第 2 节）。
2. **方法**：一个三机制的极简 agent 循环——反馈通道（固定卡 / 自建探针）、认证停止、残差签名演化引导。不使用种群、岛屿、LLM 评委或文献检索。
3. **工具的生成式校验**：agent 自建探针的信任机制——植入信号夹具（残差类：周期成分、跨通道滞后、非线性阈值注入；激活类：已知层的秩一注入）+ 方向性 / 相对裕度 / 特异性（无信号负对照）校验，通过者入库、逐轮重跑。这是“生成 → 确定性校验”在诊断工具上的实例。
4. **测试平台与证据**：一组最优预测器可精确计算的合成任务族，以及直接检验四件事的评测协议：证书是否准确、哪种反馈有用、停止是否正确、自建工具是否胜过固定工具。

---

## 2. 为什么是时序：同一原理，唯有时序零成本成立

**普遍原理（我们不声称独有）。** 设 $\mathcal{I}$ 为预测时刻的信息集，$\mathcal{G}$ 为一类 $\mathcal{I}$-可测的校正器。如果存在 $g\in\mathcal{G}$ 使 $m+g$ 的误差更小，那么 $m$ 不是最优的。这一原理在其他领域早有对应：

| 领域 | 对应形式 |
|---|---|
| 监督学习 | boosting；multicalibration / outcome indistinguishability（Hébert-Johnson et al. 2018；Globus-Harris et al. 2023） |
| 概率预测 | Brier 分数 Murphy 分解中的 reliability 项 |
| 强化学习 | Bellman 残差 |
| 预测理性检验 | Mincer–Zarnowitz；Patton & Timmermann 2012 |
| 符号回归的 LLM 演化 | Deliberate Evolution 的残差工具 |

这个原理在多数领域用不起来，原因不在原理本身，而在三个前提条件。时序预测是三者**同时、零成本**满足的领域：

| 前提 | 时序预测 | iid 监督学习（含表格、CV、符号回归） |
|---|---|---|
| **T1 规范的审计输入** | 残差自身的过去 $r_{t-k}$ 就属于 $\mathcal{I}$，是一个与任务无关的通用审计输入；隐层激活同样免训练可得（一次前向） | 其他样本的残差对当前样本没有信息，审计器类必须人为挑选。例如 Deliberate Evolution 要手工列出 $\sin x$、$\exp(-x)$ 等变换，multicalibration 要预先指定子群类 |
| **T2 闭式的线性 headroom** | Kolmogorov–Szegő 公式直接由残差谱给出一步线性 headroom：$\mu^2+\sigma_r^2-\sigma^2_{\mathrm{KS}}$，不需要训练任何审计模型 | 一般没有闭式解，需要拟合审计器 |
| **T3 能对应到结构的坐标** | headroom 可以沿频率、滞后、步长、通道分解，每个坐标都能对应到一类结构机制（下表） | 切片维度依赖人工设定的属性，和结构组件之间没有约定俗成的对应 |

**T1 的诚实补注（v0.4）。** Deliberate Evolution 的批评——“审计器类必须人为挑选”——同样适用于我们的自建探针臂：agent 也在挑自己的审计类。时序里真正免费且通用的不是探针本身，而是三样东西：(i) **审计输入**（残差过去与隐层激活，无需任何设计）；(ii) **审计坐标**（频率 / 滞后 / 步长 / 通道，T3）；(iii) **可生成的校验协议**——任何时序都能注入已知信号构造出“有信号 / 无信号”夹具对，让探针的检出方向、裕度与特异性（对无信号对照不误报）被确定性检验。iid 场景缺的正是 (i) 与 (iii)：没有免费的审计输入，也没有天然的注入-对照构造。因此我们的主张精确化为：**时序中“可审计性”是免费的，工具形态则交给 agent，由校验把关。**

| 残差坐标 | 典型缺陷 | 对应的结构机制 |
|---|---|---|
| 频率 | 周期 24 附近的谱峰残留 | 季节 / 频域模块、滤波器组 |
| 滞后 | 滞后大于 lookback 的依赖残留 | 感受野、lookback、记忆 |
| 预测步长 | 长步长上的 headroom 集中 | 解码器、直接预测与递归预测、多步一致性 |
| 通道 | 通道 $i$ 的残差可由通道 $j$ 的过去预测 | 通道独立与通道混合 |

**附加条件：最优预测器可以精确计算的非平凡任务。** 线性高斯状态空间模型（Kalman 滤波）和隐马尔可夫体制切换（前向滤波）的最优预测和最优误差都可以精确计算，并且可以构造多尺度、部分观测、周期长于 lookback、跨通道变滞后等“标准结构会系统性失败”的情形。这让“离最优还有多远”能被直接测量。其他领域也能构造最优已知的任务（如高斯混合分类），但很难同时具备 T1–T3。

> 一句话：残差可预测即非最优，这一点处处成立；**只有在时序里，审计输入、线性闭式解和结构坐标三者都是免费的**。

---

## 3. 方法

### 3.1 记号与数据切分

- 预测起点为 $o$，步长为 $h$，目标时刻 $t=o+h$，残差 $r^{(h)}_t=x_t-\hat x^{(h)}_t$；等价地，把起点为 $o'$、步长为 $h$ 的预测残差记作 $r[o',h]\equiv r^{(h)}_{o'+h}$。
- **步长与 lookback**：主实验取 $H=24$、$L=96$；$H=96$ 只作为补充设置。较短的 $H$ 计算量更小、可用的评估起点更多，也更贴近 $h=1$ 处的闭式理论。
- **信息集** $\mathcal{I}_o$ 包含：所有 $s\le o$ 的观测值（全部通道），以及截至 $o$ 已经实现的**全部跨步长残差**。滞后一律**相对起点 $o$** 计数：$r[o',h]$ 属于 $\mathcal{I}_o$ 当且仅当 $o'+h\le o$。特别地，每个步长 $h$ 的**同步长已实现残差** $r[o-h-k,h]$（$k\ge0$，目标时刻为 $o-k$）都在信息集内。信息集可以超出模型的 lookback $L$；如果超出部分有 headroom，就落在“滞后 > $L$”这个坐标上，意味着 lookback 不足。这本身也是一种缺陷。
- **数据切分**：train / diag / cert / test。
  - **diag**：agent 可见，用于生成反馈和判断是否接受候选。内部再按时间分成 diag-A（拟合校正器）和 diag-B（评估）。
  - **cert**：agent 不可见，只用于停止检验。
  - **test**：只在最后评测一次。

### 3.2 残差 headroom

**定义。** 给定校正器类 $\mathcal{G}$（其成员只使用 $\mathcal{I}_o$ 中的信息，且 $0\in\mathcal{G}$），定义

$$H_{\mathcal{G}}(m)=\mathrm{MSE}(m)-\inf_{g\in\mathcal{G}}\mathrm{MSE}(m+g).$$

**命题 1（可实现下界）。** 设 $\mathrm{MSE}^\star_h(\mathcal{I})$ 为基于 $\mathcal{I}$ 的最优 $h$ 步误差，则

$$0\le H_{\mathcal{G}}(m)\le \mathrm{MSE}(m)-\mathrm{MSE}^\star_h(\mathcal{I}).$$

当 $\mathcal{G}$ 为“残差全部过去的线性滤波”、$h=1$、残差平稳时，$H_{\mathcal{G}}=\mu^2+\sigma_r^2-\sigma^2_{\mathrm{KS}}(r)$，为闭式。

*证明思路。* $m+g$ 只使用 $\mathcal{I}$ 中的信息，所以它的误差不低于最优误差；又因为 $0\in\mathcal{G}$，所以 $H_{\mathcal{G}}\ge0$。闭式部分由 Kolmogorov–Szegő 定理直接得到。

**有限样本的诚实版本。** 在 diag-A 上拟合 $\hat g$，在 diag-B（或 cert）上计算实现增益

$$\hat\Delta=\frac{1}{|B|}\sum_{t\in B}\big[r_t^2-(r_t-\hat g_t)^2\big].$$

在 A、B 两段分布一致（平稳）的条件下，$\mathbb{E}[\hat\Delta\mid\hat g]=\mathrm{MSE}(m)-\mathrm{MSE}(m+\hat g)\le H_{\mathcal{G}}\le$ 到最优的差距。所以 $\hat\Delta$ 的**置信下界**是到最优差距的置信下界，也就是一个带置信度的“非最优性见证”。

置信界用 HAC 方差（Newey–West，滞后不小于 $h-1$）加 HLN 小样本修正解析地计算，以处理重叠窗口带来的相关性；缺陷卡各行即**点估计 + 解析置信下界**。

> 这里刻意以“实现增益”作为主量，而不是 plug-in 的谱估计：后者在有限样本下会高估可实现的修正量。

### 3.3 反馈：缺陷卡（固定工具臂）

对每个坐标 $c$，只用该坐标对应的输入拟合一个**线性**校正器 $\hat g_c$，报告它的实现增益 $\hat\Delta_c$ 及置信下界：

| 坐标 | 校正器输入 |
|---|---|
| 频段 $b$ | 过去残差经**因果**带通滤波后的分量 |
| 滞后段 $[k_1,k_2]$ | 这些滞后上已经实现的残差（$k\ge h$） |
| 通道 $j\to i$ | 通道 $j$ 的过去残差和观测值。通道很多时，先用多变量 Szegő 的 log-det 形式筛出总量，再按 $\hat\Delta$ 只展开前几个 |
| 步长组 | 上面各项按 $h$ 分组分别报告 |

**性质。** 每个 $\hat\Delta_c$ 本身都是可以兑现的增益，因此都是到最优差距的下界。各坐标之间**不要求可加**，卡片只按置信下界排序，不报告“占比”。

线性一步 headroom 的闭式值，以及谱的 Itakura–Saito 密度 $\sigma^2_{\mathrm{KS}}(x-1-\log x)$（其中 $x=f_r/\sigma^2_{\mathrm{KS}}$，逐点非负），只作为可视化附在卡片里，不参与判定。估计时使用 multitaper，并剔除 $\omega=0$ 和 $\omega=\pi$ 两个端点。

示例：

```
DEFECT CARD   cand-07   diag MSE=0.412
  total realizable headroom (all coords)  Δ=0.061  LCB95=0.047
  #1 band period≈24      Δ=0.034 LCB=0.026   strongest at h∈[1,24], ch3/ch7
  #2 lag 200–340 (>L=96) Δ=0.015 LCB=0.009   → information beyond lookback
  #3 ch2 → ch5           Δ=0.006 LCB=0.002
```

**认证结果不出现在卡片上**：认证只在后台的 cert 段上计算，写入 agent 不可见的记录；停止点由后台判定与执行（实现上所有运行都跑满预算，认证停止点由后台事后精确读出）。**v0.4 修正**：证书的**饱和比特**会作为演化引导的触发器（§3.7）以措辞中性、无数值的形式进入所有臂的提示；除此之外 agent 收到的每一步反馈都不含认证信息。

### 3.4 循环

评估、接受与认证都由循环在 **agent 之外**执行。agent **没有 bash**（只有读写工作区的文件工具）：每轮它基于当前 incumbent 的代码和反馈，写出**恰好一个**候选 `model.py` 和一份 `proposal.json`，不能自己运行评估、拟合校正器或读取 diag 之外的数据。训练、评估、接受判定和认证全部由后端确定性执行。

**初始 incumbent 为 PatchTST-lite**（patch 16 / stride 8，$d_{model}=64$，2 个 pre-norm block，通道独立，CPU 数秒可训）：隐层分析因此作用于真实 transformer 激活，而非线性映射；深度结构也给了结构演化真实的空间。

```
父模型 m ──► 反馈（缺陷卡 / 自建探针输出）──► LLM 写出“目标坐标 + 预期”并修改代码 ──► 循环训练并评估候选 m'
   ▲                                                                        │
   └──── diag 上 MSE 显著下降（DM 检验 + HAC）则接受，否则保留 m ◄──────────┘
incumbent 改变时：后台在 cert 上检验 ε-饱和 → 记录停止点（数值对 agent 不可见）
证书饱和 或 连续接受集中于同一坐标族 ──► 多样性切换：改推签名不相交的最佳历史候选为起点（§3.7）
```

- **接受规则**：所有对比组相同，只看 diag 上 MSE 是否显著下降。
- **LLM 的输出格式**：所有组都必须写出目标坐标和预期，以便统一计算“缺陷定向率”。
- **自建探针臂**的 agent 还可以写探针（§3.6）；探针校验不占候选预算，不影响接受流程。

### 3.5 认证停止

**规则（一句话）。** 每当 incumbent 更换，在 agent 看不见的 cert 段上检查：每个统计量（总量、各坐标、各步长组）的可实现增益 $95\%$ 上置信界是否**都**低于 $\varepsilon\cdot$ 该项 MSE——是则记录停止点。

没有第二句：固定水平、无跨版本预算、无族摊销（修订 II 删除了原 α 记帐与命题 2）。置信区间按 Newey–West + HLN 解析计算；区间是双侧的，因此每个 UCB 实际是单侧 $1-\alpha/2$ 界，只在让停止更保守的方向上偏。聚合统计量（total）用贪心坐标选择构造——它仍是已实现信息集上的线性校正器，命题 1 的下界不受影响。

**误停是实测性质，不是解析保证。** 我们不主张停止规则的误停概率界；负对照任务（GARCH、随机游走）与植入缺陷模型上的误停率由 G0-1/G0-5 直接测量并随论文报告。认证日志只是一份记录，不是预算账本。

**诚实的边界。** 这个判据只针对校正器类 $\mathcal{G}$ 和信息集 $\mathcal{I}$。它是“$\mathcal{G}$-相对的 ε-饱和”，**不是**贝叶斯最优的证书。对固定的有限维线性 $\mathcal{G}$，$\Delta$ 与 $H_{\mathcal{G}}$ 之间的差距随样本量增加而缩小。

**证书与搜索路径的关系（v0.4 修正）。** cert 的**数值**对 agent 完全不可见（§3.3），实现上所有运行都跑满预算、由后台事后读出停止点。但方向修订把**饱和比特**交给了演化引导（§3.7，三臂一致）：因此“incumbent 序列与 cert 无关”不再成立到比特级——第 $j$ 次认证的饱和状态会影响第 $j$ 轮之后的起点推荐。我们如实表述为：**证书在数值上不可见，在饱和比特上以措辞中性的形式进入所有臂**；RQ3 的“停止后继续搜索的收益”因此是在主动多样性协议下测得的，报告时明确标注（若需要“朴素继续”的停后收益，可用 `evolution=false` 消融运行）。

### 3.6 自建探针：工具的生成式校验（新主线）

自建臂的 agent 可写探针：`probes/<id>/probe.py` + 提案（声明探针测什么、其主输出键、期望方向、以及它读取残差还是激活）。探针只拿到两样东西——incumbent 在 diag 段的**残差序列**（1 步已实现序列 + 按 $h$ 的残差矩阵）与**逐层激活**（256 个均匀采样窗口，与残差行按索引对齐，可回答“哪些内部状态先于哪些误差”）；拿不到原始窗口、目标值或任何 cert/test 数据，探针输出（≤4KB）经统一渲染回灌给 agent，构成本臂唯一的中段反馈。

**校验协议（信任机制）。** 新探针先在**植入已知信号的合成夹具**上接受检验，通过才入库、才有资格在真实数据上运行：
- 残差类夹具族：周期成分（如周期 24 正弦，2 对）、跨通道滞后注入（2 对）、**非线性阈值注入**（2 对——线性探针必须检不出，非线性探针必须检出）；每对为“植入 / 干净”配对，另加**无信号负对照**（同分布新种子）。
- 激活类夹具族：固定小网络上、已知层处的秩一注入（2 对，与残差的时间成分对齐）。
- 通过规则：至少**完整通过一个族**（该族每对：方向正确、相对裕度 > 5% 对干净成员与全部负对照成立、且植入值对无信号簇满足非参数的“间隔大于簇宽”底线），并满足确定性（同夹具重跑输出一致）与 4KB 上限。诚实噪声底：重尾近零探针在可交换性下仍有 ~1–7% 的单次校验误过率（蒙特卡洛实测，文档化）。

这一“生成 → 确定性校验”的信任机制是探针通道的核心：探针不是因为看起来合理而被信任，而是因为在植入信号上可证地检出且不误报。**诚实披露**：该臂相对固定卡臂捆绑了两个差异——工具由谁构建，以及工具可读激活通道（固定卡只有残差坐标）；主对比测量的是这个合并处理。首轮自举延迟（首个探针通过校验前该臂只有标量反馈）与渲染上限（最新 8 个探针、每行 600 字符）作为处理的一部分在附录 D 中披露。

### 3.7 残差签名演化引导（时序结合的多样性机制）

每个已评估候选获得一个**残差坐标签名**：相对其父代，它关闭了哪些坐标族（来自后台事后卡片；“关闭”= 父代该坐标 LCB>0 而候选 ≤0；若无完全关闭，取 LCB 下降最大者）。这构成一个按“修好了什么”而非“基因型”索引的候选档案——多样性轴就是证书测量的时序坐标空间。

**触发与动作。** 当（a）最新一次认证判饱和，或（b）连续 3 次接受都命中同一坐标族时，本轮提示改为**推荐签名与当前坐标族不相交的最佳已评估候选**为起点（复制其代码再修改；接受仍按 DM 对 incumbent 判定）。措辞对所有臂**完全一致**且不含任何坐标名或 heldout 数值。机制由证书触发是设计的一部分（方向修订 §12.3）：饱和意味着“沿这条线已无可兑现的改进”，正确的反应是换方向，而不是继续磨。

- 消融开关：`evolution=false` 供停后收益的“朴素继续”对照。
- 诊断量：切换次数；每次切换后首个被接受候选的签名是否与切换前坐标族**不相交**（postSwitchDiversity）。

---

## 4. 实验设计

### 4.1 研究问题

- **RQ1 证书准吗？** 在最优已知的任务上，置信下界是否始终不超过真实差距？估计的 headroom 和真实差距的相关性有多高？误停率是多少？这一组**不涉及 LLM**。
- **RQ2 哪种反馈有用？** 相同预算下：**agent 自建并经植入信号校验的探针（含激活通道）能否胜过后端固定缺陷卡？固定卡能否胜过标量？**（自建 vs 固定为主对比；激活通道的单独贡献留作消融。）
- **RQ3 停得对吗？** 停止时剩余的真实差距是多少？在主动多样性协议下、停止点之后继续搜索还有多少收益？（`evolution=false` 消融给出朴素继续版本。）
- **RQ4 LLM 改结构有必要吗？** 相比直接把残差校正器叠加上去（$m+g$），LLM 修改结构能多带来什么？

### 4.2 任务

**合成任务（最优误差精确已知）**

| 过程 | 用来制造的结构性困难 |
|---|---|
| 线性高斯状态空间（Kalman 最优） | 多尺度周期、周期长于 lookback、部分观测、跨通道变滞后耦合 |
| HMM 体制切换（前向滤波最优） | 非平稳的均值和动态 |
| ARFIMA | 长记忆 |
| **负对照**：GARCH、随机游走 | 均值不可预测，证书应在第 0 轮就判定饱和 |

**切分。** 合成任务的 train 与 diag 来自同一条主序列；cert 与 test 是**同一过程的一次全新独立模拟**（长度分别不少于 $2\times10^4$ 与 $5\times10^4$ 步，带 burn-in），与主序列不共享任何数据点。真实任务仍按时间顺序切成 train / diag / cert / test。

**oracle。** 最优参照定义为**全信息最优**：oracle 使用截至起点的完整历史，信息深度 $\ge D$，其中 $D=\text{lag 上限}+\text{FIR 长度}$ 覆盖任何校正器可能用到的最深输入，因此没有校正器能改进 oracle。实现上，Kalman 与 HMM 任务从序列起点起跑滤波再做 $h$ 步外推；ARFIMA 用 Durbin–Levinson 递推拟合高阶 AR（阶数 2000）。gap 定义为候选与 oracle 在**同一批起点**上配对计算的经验 MSE 之差。“L-受限 oracle”和“正确设定的模型在同一训练集上拟合后的误差”只作为次要参照；后者用来扣除有限样本带来的估计误差。

**真实任务**
- 去标识化的 ETTh1 为主（M4 评审后从 GIFT-Eval 子集收敛到单一可复现真实任务；Exchange-Rate 作为负对照，卡片应报无 headroom）。
- 数据集名称和领域描述**去标识化**，避免 LLM 直接调用它对这些公开数据集的记忆。
- 所有候选的 lookback、归一化和调参预算保持一致。

### 4.3 对比组

**析因主实验**：同一个 LLM，所有组共享认证停止与演化引导，都写预期、都用 DM 接受规则。臂间差异只在反馈通道。

| 组 | 反馈通道（臂间唯一差异） |
|---|---|
| Scalar | diag MSE、按步长分组的 MSE 表与修改历史（+ 共享引导） |
| Fixed | 外加后端**缺陷卡**（§3.3；原 Card 臂，更名以对称于自建） |
| **Self-built** | agent 自建探针的校验结论与输出（§3.6；含激活通道）——无固定卡内容注入 |
| Placebo Card（补充设置） | 打乱坐标行的卡片（prompt 层对照，不进主矩阵） |

- Scalar+BJ 臂退役：M4 评审确认固定卡与 BJ 的对比可由“固定 vs 标量”承载，新主线是自建 vs 固定。
- **公平性披露**：自建臂首轮无探针输出（自举延迟是处理的一部分）；探针校验不占候选预算；每臂 token 与墙上时间单独核算。固定卡的措辞与自建探针指南同为实验工具，冻结后原文进附录。

**非 LLM 基线**
- $m_0+g$ 迭代残差 boosting（$m_0$ 同为 PatchTST-lite，与主实验同起点）；
- 规则修复：按坐标套用固定修法；
- auto.arima、ETS、状态空间模型。

**相关系统**
- EvoCast：“EvoCast + 缺陷卡”留作未来工作，用来日后验证缺陷卡与搜索框架的正交性。
- SEATS（复现其中可行的部分）。

**公平性处理**
- 所有方法的最终模型都统一再做一次 $m+g$ 后处理，然后再比较。这样可以扣除“能被直接兑现的那部分 headroom”，看剩下的结构性收益。
- 预算横轴同时给出候选数、墙上时间和 LLM token 数。

### 4.4 指标

- **最优性差距**：$(\mathrm{MSE}-\mathrm{MSE}^\star)/\mathrm{MSE}^\star$ 随预算的变化曲线（合成任务）。
- **主终点（预先登记）**：统一 $m+g$ 后处理之后、第 25 个候选处的归一化 test 最优性差距（真实任务报告相对 MSE 比），以及前 25 个候选的差距曲线下面积（AUC）。
- **证书校准**：置信下界与真实差距的散点图；误停率和漏停率。
- **缺陷定向率**：对所有组的**全部提案**，事后计算修改是否缩小了修改前的头号缺陷。
- **off-target 率**：MSE 下降、但声称的目标坐标没有缩小的比例。
- **演化诊断**：切换次数；切换后首个接受是否落在不相交坐标族（postSwitchDiversity）。
- **探针账本**（自建臂）：写入 / 通过校验 / 入库数量，探针渲染覆盖率，探验耗时。
- **diag 与 test 增益的相关性**、成本统计。
- **统计协议**：每格（任务 × 臂）**8 次独立运行**，报告 IQR。臂间比较以 run 为单位，做**按任务分层、按 (task, seed) 块的置换检验**（主对比 self-built vs fixed 共 32 个块，对 ~0.5 SD 的配对效应有 ~80% 功效）；DM（HAC 方差）只用于同一批起点上两组预测的直接比较，即 §3.4 的接受规则。

### 4.5 主图

**图 1**：在一个最优已知的合成任务上，横轴为预算，纵轴为真实最优性差距。画出各臂曲线，叠加固定卡臂估计的 headroom 及其置信带，标出认证停止点与多样性切换事件。

这一张图同时回答 RQ1–RQ3。

### 4.6 消融

| 消融 | 检验什么 |
|---|---|
| `evolution=false` | 演化引导的贡献；停后收益的“朴素继续”版本 |
| 自建臂关掉激活通道（探针只拿残差） | 激活通道在主对比中的单独贡献 |
| 只保留线性校正器类 / 加入非线性校正器 | 校正器类 $\mathcal{G}$ 的作用 |
| 去掉置信界，只给点估计 | 统计语义的作用 |
| 探针校验去掉负对照 | 特异性检验的作用（预期噪声探针混入） |
| 开放 bash + 可读 train/diag | 强 agent 自己做诊断是否更好（主循环中 agent 无 bash，见 §3.4） |
| 换用不同的 LLM | 结论是否依赖特定 LLM |

领域坐标（§3.3）与“EvoCast + 缺陷卡”不进本文实验，留作未来工作。

---

## 5. 相关工作与差异

| 工作 | 反馈 / 诊断 | 最优性语义 | 停止 | 与本文的关系 |
|---|---|---|---|---|
| **EvoCast**（2610.04517） | TFB 数据特征和对父模型机制做**执行消融**的 Δ；带反局部延续约束 | 无 | 20 轮预算 | 最接近的时序结构演化系统。它的诊断是“由内向外”：模型里已有的组件哪些有用；我们的诊断是“由外向内”：数据里还有什么没被模型吃掉、还剩多少。其反局部延续约束是我们签名档案多样性切换的近亲，但我们的签名来自带置信界的残差坐标，且由证书饱和触发——“该停了”与“该换方向”由同一个量驱动 |
| **Deliberate Evolution**（2606.04360） | 残差与**人工挑选**的变换的相关；iid 符号回归 | 无 | 固定预算或停滞检测 | 同一普遍原理在 iid 场景的手工版本，正好说明 T1（见 §2 的诚实补注）。它的“假设引导优势 γ”在我们的设置里可由缺陷定向率直接测量；我们的自建探针把“挑选审计类”交给 agent，用生成式校验替代人工挑选 |
| CTRL（2609.23257） | 把残差的 STL 分解统计交给 LLM | 无 | — | 修正的是输出，不改结构 |
| MARICL（2605.22897） | 从表格数据的残差模式生成假设 | 无 | — | iid 版本 |
| OPERA、Little Scientist、Self-EvolveRec、Component-Aware Feedback | 结构化或逐实例反馈 | 无 | 预算 | 证明结构化反馈普遍有效。我们的增量在于最优性语义、认证停止与校验过的自建工具 |
| SEATS、TS-Agent、TimeSeriesScientist、FLAIRR-TS | 标量指标；FLAIRR-TS 在增益低于阈值时停止 | 无 | 预算 / 阈值 | 标量反馈的代表 |
| Automatic Statistician（Lloyd et al. 2014） | 残差和结构驱动的 GP 核组合搜索 | 无 | 模型证据 | 诊断驱动结构发现的先驱，局限在 GP 文法内 |
| CRAFTER（2608.05207）、Sun et al. NeurIPS 2021、ARIMA-ANN 混合（Zhang 2003） | 对残差建模并叠加校正 | 无 | — | 就是我们的 $m+g$ 基线 |
| ForeCA、加权排列熵、谱可预测性 Ω、Spectral Coherence Predictability（2509.23074） | 数据本身的可预测性指标 | 针对数据，而非针对模型残差的 headroom | — | 衡量的是“数据有多可预测”，不是“这个模型还差多少” |
| multicalibration / outcome indistinguishability、Patton–Timmermann | 审计器与最优性检验 | 有 | — | 同一普遍原理（第 2 节），不面向结构搜索 |

**本文能守住的新颖性**：(i) 把残差 headroom 作为 LLM 结构搜索的**反馈**，按时序坐标分解，每一项可兑现；(ii) **ε-饱和认证停止**（固定水平单句规则，误停率实证测得）；(iii) **agent 自建诊断工具的生成式校验**（植入信号夹具 + 方向 / 裕度 / 特异性规则）与“自建 vs 固定”工具对比；(iv) **残差签名演化引导**：多样性轴即证书坐标空间，由证书饱和触发；(v) **最优已知的测试平台**与直接检验证书、反馈、停止、工具四者的评测协议。

“写出可证伪预期”（EvoCast 已有）和“结构化反馈优于标量”（OPERA 等已证明）**不再**列为贡献。

---

## 6. 预期质疑与回应

| 质疑 | 回应 |
|---|---|
| 原理不是时序独有 | 同意，见第 2 节。主张精确化为“唯有时序中可审计性零成本”：审计输入与校验协议免费，工具形态交给 agent |
| 自建臂也在“人为挑选审计类”，与 Deliberate Evolution 有何不同 | 不同在信任机制：探针要在植入信号夹具上通过方向 / 裕度 / 特异性校验才能成为反馈，挑选错误会被校验拒绝（M4 评审实测：纯噪声探针在负对照规则下出局）。iid 场景没有等价的免费夹具构造 |
| 不就是 Ljung–Box 加 prompt 吗 | 自建臂正面检验这一点：若 agent 只会复刻 BJ 统计量，它相对固定卡不会有增量；校验协议还要求它至少完整检出一个植入族（含非线性族） |
| $m+g$ 就够了，为什么要 LLM | RQ4 正面检验，并且所有最终模型统一做 $m+g$ 后处理再比较 |
| 饱和不等于最优 | 同意。判据只声称 $\mathcal{G}$-相对的 ε-饱和；合成任务上报告停止时的真实剩余差距，负对照上的误停率由 G0 实测 |
| 饱和比特进入了提示，证书还“不可见”吗 | 数值不可见；饱和比特以中性措辞进入所有臂（§3.5 如实修正），RQ3 的停后收益在主动多样性协议下报告，`evolution=false` 消融给出朴素版本 |
| 非平稳数据会让下界失效 | 下界要求 diag 与目标期分布一致。我们报告 diag 与 test 的相关性，并单独列出均值偏移项 $\mu^2$，与结构性缺陷区分开 |
| LLM 记得公开数据集 | 数据集去标识化，并以合成任务为主证据 |

---

## 7. 最小验证实验

**阶段 0：证书校准（G0），不需要 LLM。**
在合成任务与负对照上，用深度 m0 及其**缺陷变体**（无注意力、粗 patch、通道混淆）检验三件事：
1. 置信下界不超过真实差距（模型无关门沿 v1 传递；紧度 / 定位 / 筛选三门用深度变体重校准，冻结 `g0-final-report-v2`）；
2. 头号坐标与植入的缺陷一致；
3. 负对照在第 0 轮就判定饱和。

**如果这一步不成立，后面的实验都不必做。** 批处理驱动拒绝在 v2 G0 工件冻结前启动任何 v2 单元。

**阶段 1：循环就绪后直接跑析因主实验。**
主实验即 §4.3：Scalar / Fixed / **Self-built** 三臂 + $m_0+g$ 基线；3 个合成任务加 1 个真实任务（去标识化）；每格 8 次运行、25 个候选；全部最终模型统一做 $m+g$ 后处理。启动门槛（G1 式冒烟）：自建臂无人值守 10 候选，失败率 < 30%、至少一个探针通过校验并被渲染。

成功信号：
- 自建或固定臂的最优性差距曲线显著低于标量臂（统一 $m+g$ 后处理后优势仍在）；
- 自建 vs 固定的主对比有方向明确的结论（无论方向如何，附探验账本与 token 对齐的次级比较）。

---

## 附录 A：与 RecursiveECG 底座的对应

| 组件 | 复用的现有能力 |
|---|---|
| 任务与数据契约 | `problem-contract` |
| 探针的生成与校验 | `knowledge-to-tools` 的“生成 → 确定性校验”机制（§3.6） |
| headroom、缺陷卡、接受与停止检验 | 后端确定性实现，agent 只能调用 |
| 修改循环 | `iterative-solving`；停止条件为认证停止；起点推荐为残差签名演化（§3.7） |
| 证据链 | `final-summary` 与现有的 run / record 机制 |

## 附录 B：参考文献（待核对原文后补全）

EvoCast arXiv:2610.04517；Deliberate Evolution arXiv:2606.04360；CTRL arXiv:2609.23257；MARICL arXiv:2605.22897；CRAFTER arXiv:2608.05207；OPERA arXiv:2608.05990；The Little Scientist arXiv:2608.16951；Self-EvolveRec arXiv:2602.12612；Component-Aware Feedback arXiv:2609.38639；SEATS arXiv:2603.04873；TimeSeriesScientist arXiv:2510.01538；TS-Agent arXiv:2508.13915；FLAIRR-TS (EMNLP Findings 2025)；TimeRecipe arXiv:2506.06482；2512.22702（ICML 2026 benchmarking position）；Spectral Coherence Predictability arXiv:2509.23074；Lloyd et al. Automatic Statistician (AAAI 2014)；Goerg ForeCA (ICML 2013)；Garland et al. PRE 2014；Sun et al. NeurIPS 2021 arXiv:2101.12578；Zhang 2003 Neurocomputing；Hébert-Johnson et al. ICML 2018；Globus-Harris et al. ICML 2023；Patton & Timmermann JBES 2012；Box & Jenkins 1970；Ljung & Box 1978；Kolmogorov 1941；Szegő 1920；Whittle 1963；Politis & Romano 1994；Diebold & Mariano 1995；Levinson 1947；Durbin 1960（Durbin–Levinson 递推）。

## 附录 C：v0.3 相对 v0.1 的变更

1. 头条从“打破固着”改为“缺什么、在哪、何时停”。固着相关内容降为分析指标（缺陷定向率、off-target 率）。
2. 第 2 节承认原理的普遍性，改为“T1–T3 零成本共存”的论证。
3. “最优性证书 / 已认证饱和”改为“非最优性见证 + $\mathcal{G}$-相对 ε-饱和”，新增命题 1、命题 2 和观察 0。
4. headroom 主量改为 held-out 的实现增益；频段分解改为逐坐标可兑现的增益，不再声称可加。
5. 信息集包含全部已实现的跨步长残差，lookback 不足作为一个显式坐标。
6. 主方法删除逐层定位（移到附录消融；v0.4 经方向修订回归主线，见 §3.6），所有组都写预期。
7. 合成任务改为 Kalman / HMM 等最优误差精确可算的过程，新增非线性白噪声过程和负对照。
8. 新增基线与安慰剂卡片；所有最终模型统一做 $m+g$ 后处理。
9. 统计协议：DM 检验 + HAC、block bootstrap、cert 切分。
10. 相关工作新增 EvoCast、Deliberate Evolution、CTRL、MARICL、CRAFTER、multicalibration 等，并重新界定能守住的新颖性。

## 附录 D：探针契约原文（实验工具冻结记录）

以下为自建臂每轮注入的探针指南原文（`experiments/prompts/probes.md`），与缺陷卡措辞同属冻结的实验工具；渲染规则：最新 8 个探针、每行输出截断至 600 字符；校验结论（kept/rejected）与输出同时呈现。

```markdown
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
  new one under a new id. Old probes keep running; each round shows the
  newest 8. Editing a validated probe's file invalidates it — the harness
  detects the change and refuses to run it.

## probe.py contract

The file must define `create_probe()`, returning an object with:

- `probe_id: str` — must equal the probe directory name;
- `run(artifacts: dict) -> dict`: returns your diagnostic summary — a small
  JSON-serializable dict of at most a few KB. Put the numbers you want to
  watch every round in it.

`artifacts` contains, computed from the CURRENT incumbent on the diagnostic
segment:

- `residuals["e"]`: `[T, C]` float array — the incumbent's realized 1-step
  forecast-error series (its own past mistakes; NaN-padded at head and tail).
- `residuals["r"]`: `[n, H, C]` float array — the incumbent's H-step residuals
  per forecast origin, aligned with `residuals["origins"]` (time indices).
- `activations`: dict mapping layer name to a `[m, P, D]` float32 array of the
  incumbent's hidden-layer activations on `m` evenly spaced diagnostic
  windows. `activations[name][i]` belongs to the same forecast as residual row
  `meta["activationOrigins"][i]` of `residuals["r"]` — those are ROW indices
  into `r`, not time indices; the forecast's time index is
  `residuals["origins"][meta["activationOrigins"][i]]`. So you can ask which
  internal states precede which errors. Empty dict if the incumbent has no
  activation path.
- `meta`: `{"L", "H", "C", "T", "seed", "incumbent", "task", "activationOrigins"}`.

Allowed imports: `numpy`, `math`, `scipy`, `torch` (CPU). No file, network, or
system access.
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
fixtures with KNOWN planted structure — three residual families (a periodic
component, a cross-channel lag injection, and a nonlinear threshold coupling,
each at two signal draws and undisclosed parameters) and one activation
family (planted structure at a known layer of a fixed small network, two
draws). Every fixture pair also carries clean-only negative controls at
fresh seeds. Your probe is kept only if, for at least ONE complete family:
on every planted pair its `signalKey` number moves in the declared direction
against the clean member AND every control, with a clear relative margin,
and the planted value stands clearly above the whole clean cluster; plus the
probe is deterministic (same fixture twice, identical output) and within the
size cap. The verdict appears in your next round's feedback. A validated
probe is re-run on every future incumbent and its output is shown to you
every round.

A probe that times out (120 s), crashes, exceeds the output cap, or returns
nondeterministic results is rejected with the reason in your feedback. Fix it
under a new id.
```
