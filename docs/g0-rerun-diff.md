# G0 重测与旧结果的差异（2026-10-19）

重测在提交 c123500 上进行，环境记录在 `experiments/runs/stage0/ENVIRONMENT.md`（Python 3.11.16、torch 2.14.1、numpy 2.4.6、Apple M4 10 核 16 GB），不调用任何 LLM。线性门控（七个门控）用时约 10 分钟，深度门控用时约 39 分钟，两者并行运行；CAL-1、2、3 重跑用时约 15 秒。

旧结果另存在：

- `experiments/specs/g0-final-report-v3-prefix-20261016.{json,md}`、`experiments/specs/g0-final-report-v3-deep-prefix-20261016.{json,md}`（旧的冻结报告）
- `experiments/runs/stage0-prefix-20261016`、`experiments/runs/stage0-v2-prefix-20261016`、`experiments/runs/calibrations-prefix-20261016`（旧的原始运行输出）

新冻结报告覆盖 `experiments/specs/g0-final-report-v3.{json,md}` 与 `g0-final-report-v3-deep.{json,md}`，`main-v3.json` 的引用路径不变。

## 修复对这些门控的影响

两处修复进入修正器的拟合：贪心选择的候选加入截距（`_val_mse` 里减去 `y_fit.mean`），通道坐标的 lag 窗口由 25 个改为 24 个（k = 0..23）。其余修复（反馈渲染、台账恢复、判官触发、探针冻结副本）不在 G0 的路径上。

## 线性门控（G0-1、2、3+4、5、6、7）

| 门控 | 判定 | 旧 | 新 | 说明 |
|---|---|---|---|---|
| G0-1 oracle 负对照 | PASS → PASS | 18/18 个 cell 的 total LCB < 0，18/18 判饱和 | 完全相同 | 个别 cell 的 total LCB 数值有变化（例如第 7 个 −0.0331 → −0.0189），符号全部不变 |
| G0-2 CI 校准 | PASS → PASS | LCB/UCB 覆盖率 1.0/1.0（三个任务） | 完全相同 | 真值/间隙比的均值 lgssm_multiperiod 0.706 → 0.708，arfima 与 hmm_ar_switch 逐位相同 |
| G0-3+4 紧致度与定位 | PASS → PASS | 紧致度中位数 0.606，定位 17/21 = 0.81 | 0.603，17/21 = 0.81 | 定位的命中数与总数相同 |
| G0-5 误停率 | PASS → PASS | 21 个 cell，1 个判饱和，真值违规 0 | 完全相同 | |
| G0-6 筛选（m0-linear） | PASS → PASS | 4 个任务合格 | 4 个任务合格 | 剩余间隙比：lgssm_multiperiod 0.291 → 0.294，nonlinear_xchannel 0.544 → 0.542，其余三个任务逐位相同 |
| G0-7 解析对照 | PASS → PASS | AR(1) 比值 1.091，白噪声 −0.0042 | 逐位相同 | 不经过修正器 |

## 深度门控（G0-3+4、G0-6，m0-patchtst）

| 门控 | 判定 | 旧 | 新 |
|---|---|---|---|
| G0-3+4 | PASS → PASS | 紧致度中位数 0.490，定位 49/54 = 0.907 | 0.491，49/54 = 0.907 |
| G0-6 | PASS → PASS | 4 个任务合格 | 4 个任务合格 |

G0-6 在 P2 任务（深度 m0 之后仍留下的剩余间隙比，阈值 0.30）上的数值：

| 任务 | 旧 | 新 | 建议 |
|---|---|---|---|
| hmm_ar_switch | 0.857 | 0.848 | keep |
| nonlinear_xchannel | 0.529 | 0.527 | keep |
| lgssm_xchannel | 0.452 | 0.450 | keep |
| lgssm_multiperiod | 0.288 | 0.291 | 不在 P2 任务列表 |
| arfima | 0.702 | 0.702 | 不在 P2 任务列表 |

## 校准 CAL-1、2、3

| 校准 | 旧 | 新 |
|---|---|---|
| CAL-1 HAC 带宽 | 冻结在 359，所有带宽覆盖率 ≥ 0.9，1200 处宽度比 1.2639 | 冻结在 359，所有带宽覆盖率 ≥ 0.9，1200 处宽度比 1.2651 |
| CAL-2 ETTh1 接受检验 | 零假设误接受 0，植入信号检出率 1.0，PASS | 完全相同 |
| CAL-3 Exchange-Rate 负对照 | 第 0 轮认证不判饱和（`saturated=false`，17 项统计量高于阈值），`pass=false` | 认证结果逐位相同（total delta −0.0140、LCB −0.0459、17 项统计量）；诊断段卡片的 total delta 由 −0.0060 变为 −0.0323（贪心选择现在带截距），LCB 与 UCB 随之变化，符号不变 |

CAL-3 的 `pass=false` 是 `main-v3.json` 里 `calibrationEvidence.cal3-exchange-rate` 已经记录的结论（认证在 735 个起点的小 cert 段上保守地拒绝判饱和），不是本次重测引入的变化。

## 结论

没有任何门控翻转，所有判定不变。数值变化最大的是 G0-1 个别 cell 的 total LCB，其次是 G0-6 的 lgssm_multiperiod（0.291 → 0.294）和 hmm_ar_switch 的深度剩余间隙比（0.857 → 0.848），方向与量级和修复内容（截距使残差均值被吸收，lag 窗口少一个）一致，属预期内的口径变化。P2 任务列表（hmm_ar_switch、nonlinear_xchannel、lgssm_xchannel）三个任务的剩余间隙比都远高于 0.30，任务列表和 G0-6 的筛选阈值不变。

重测前的失效清单（`docs/main-v3-rerun-plan.md`）把 G0-2、G0-3+4、G0-5、G0-6、CAL1-3 列为必须重测，原因是修复会改变修正器的拟合。重测的结果是：这些数值变化很小，所有判定不变，旧报告所支持的结论在新口径下依旧成立。
