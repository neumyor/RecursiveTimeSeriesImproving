# 全仓静态评审报告（2026-10-18）

按 behavior-norms 第 3 节（项目范围）和第 4 节（代码质量、禁止事项、token 预算）对整个仓库做的静态评审，只读代码，没有运行实验。四个子智能体分区阅读了全部源码。标注"已复核"的条目由主智能体亲自验证过证据；其余来自子智能体逐行阅读代码。

## 第一节：影响论文实验结论的错误（experiments/、headroom/）

1. `experiments/loop.py:162`（已复核）`_hgroup_mse` 的第二组写成了 `seg.r[half:, :, :]`。`r` 的维度是 `[B, H, C]`，这样切的是起点维度。结果是所有实验组反馈给智能体的 `hgroup[13,24]` 数值都错了。应改为 `seg.r[:, half:, :]`。
2. `backend/harnessing_ts/headroom/correctors.py:446`（已复核）贪心选择时，基线预测带有拟合段的均值，候选预测 `design_val @ coef` 却没有截距。残差均值不为零时（例如 level-bias 夹具），每个候选会多出约等于均值平方的误差。贪心选择因此少选甚至不选，认证结论偏向"已饱和"。已冻结的 G0-v3 结果可能受影响，修复后需要重新运行涉及 `total`/`hgroup` 认证的门控。
3. `experiments/batch.py:450`（已复核）抛出 `SystemicFailure` 后，`finally` 中的 `pool.shutdown(wait=True)` 没有加 `cancel_futures=True`，排队的单元格会全部继续运行。账户欠费时仍会跑完整个矩阵。
4. `experiments/loop.py:1076`（已复核）探针分支重新给 `started` 赋值，覆盖了本轮的开始时间。第 1128 行记录的 `seconds` 在 self-built 和 augmented 两组偏小，各组的耗时对比有偏差。
5. `experiments/loop.py:855`（已复核）连续 3 轮没有接受候选后，之后每一轮都会调用多样性判定。与第 112 行的设计说明和规格里的调用预算矛盾。
6. `experiments/loop.py:1142`（已复核）候选收集失败后，这个目录在本轮会被排除。智能体按修复提示在原目录修改后，循环仍然回复"没有写候选"，这次修复就浪费了。
7. `experiments/posthoc.py:726`（已复核）除法写在了 `sum` 里面，`roundsWithProbeOutput` 得到的是计数。规格要求的是比例。
8. `experiments/posthoc.py:252` 只要存在 `incumbent-reference.json`，就总是覆盖 `--m0-template`，这个参数实际不起作用。保守做法：显式参数优先。
9. `experiments/stage0.py:640` deep 运行的 G0-1/G0-5 结论是从 v1 报告继承来的。规格指定的固定水平结果是 v3。
10. `experiments/prompts/probes.md:68` 提示词写的是三个残差家族，注册表里已经有五个。提示词冻结前需要改正。
11. 中断恢复不完整：
    - `experiments/loop.py:265-322`：第 1 轮崩溃后重试会新建 heldout 目录，但不写 `incumbent-reference.json`，posthoc 无法重放 m0，这次运行的数据就丢了。
    - `experiments/loop.py:1278`：认证记录在台账之前写入。两次写入之间如果崩溃，会留下一条孤立的认证记录。
    - `experiments/posthoc.py:836`：没有跑完的运行会被当作完整运行计入统计。
12. 智能体的输入被截断（违反 4.4）：
    - 探针输出最多允许 4096 字节，`backend/harnessing_ts/headroom/feedback.py:29` 只显示 600 字符。
    - `experiments/loop.py:423` 只提供最近 10 轮历史，提示词却要求候选和全部历史候选都不同。
    - `backend/harnessing_ts/headroom/candidate.py:106` 在修复轮次只提供 1200 字符的错误输出。
13. `experiments/posthoc.py:620` 规格要求的 `judgeAccounting` 和 `postSwitchDiversity` 没有计算，各组的 token 合计也没有包含 `judgeTokens`。

### 第一节的补充条目（同样改变实验数值或判定）

14. `backend/harnessing_ts/headroom/correctors.py:14-17, 205` 注释和文档说通道坐标的 lag 窗口是 k = 0..23，代码 `np.arange(_CHANNEL_LAGS + 1)` 生成的是 0..24（25 个 lag）。代码与契约 §5 不一致，改变通道块的构造。
15. `backend/harnessing_ts/headroom/synthetic.py:19, 24, 203, 457` 模块级 `_PREFIX`/`_HORIZON` 取自默认 `HeadroomConfig()`。非默认的 `horizon` 会得到错误长度的理论 MSE 和错误尺寸的前缀。
16. `backend/harnessing_ts/headroom/feedback.py:142, 158` 探针的 id、detail、output 里出现 `|` 或换行会破坏智能体读到的 markdown 表格。
17. `experiments/fixture_gen.py:570-573` 确定性重跑的 `error` 被忽略，`assert repeat_outputs is not None` 会把本该 FAIL 的门控变成 AssertionError 崩溃。

## 第二节：交互系统的功能错误（后端核心模块、前端）

1. `backend/harnessing_ts/execution.py:75`（已复核）进入 `failed` 状态后，`begin()` 只接受 `ready`/`paused`，也没有代码把状态改回 `ready`。主会话只要失败一次就无法再接收消息。
2. `backend/harnessing_ts/node_state.py:232`（已复核）`lstrip("./")` 会去掉开头所有的 `.` 和 `/`，`../user/x.md` 因此能通过输出路径检查。
3. `backend/harnessing_ts/agent/translate.py:245`（已复核）后端发送前会去掉 `raw` 字段，前端 `frontend/src/main.ts:2789` 却依靠 `part.raw?.is_error` 判断错误，所以 SDK 的错误结果永远不会显示。另外 `main.ts:2638` 引用的图标 `CircleX` 不存在，失败的工具卡片没有图标。
4. `backend/harnessing_ts/orchestrator.py:1857` 在提醒轮次中节点完成时，会在后台任务里自动进入下一个节点。结果是留下一个没有任务的"运行中"节点，旧的运行器也没有关闭。
5. `backend/harnessing_ts/mcp/server.py:53` 命令行路径的本地后备会把工作区的 `controlMode` 改成 `manual`，而且跳过了权限检查。
6. `backend/harnessing_ts/state/workspace_store.py:996` 保存 LLM 配置时会丢掉 `maxTokens`，知识图谱、推理和链路摘要的运行器也拿不到这个值。
7. `backend/harnessing_ts/state/jsonl.py:39` 写 JSON 时没有先写临时文件再替换。写到一半崩溃会损坏 `workspace.json`，之后所有请求都返回 500。
8. `backend/harnessing_ts/server.py:565` 清除日志时没有检查节点任务是否还在运行。
9. `backend/harnessing_ts/server.py:684` `except Exception: pass` 吞掉了变体禁用等错误，接口仍然返回 `accepted: true`。
10. `backend/harnessing_ts/reference_feature_extractor.py:154` 每次发送状态快照都会在服务进程里重新执行一遍智能体生成的 `extractor.py`。
11. 前端异常处理缺失：`frontend/src/main.ts:887` `EventSource` 没有 `onerror`，服务返回非 200 后界面会一直停在"加载中"；`main.ts:1399` 设置页每次重新渲染都会覆盖用户还没保存的输入；多个按钮的处理函数没有捕获异常。
12. `backend/harnessing_ts/mcp/server.py:107` `timeout=305` 对一次完整的 LLM 查询来说太短。

## 第三节：违反 4.2 的禁止事项

哈希和指纹：`knowledge_graph.py:381,1077` 对参考文件计算 sha256；`orchestrator.py:400` 用 JSON 指纹缓存去除重复消息；`main.ts:1078` 用 FNV-1a 哈希检测 DOM 变化；`orchestrator.py:304` 用 `hmac.compare_digest` 比较不需要保密的路径；`tests/test_experiments_fixture_gen.py:59` 的摘要值注释。

重复检查：MCP 工具权限在五个地方各检查一次；租约检查了四次；`headroom/candidate.py:406` 的 `_validate_result` 重复检查子进程已检查过的形状和类型；`NonlinearXChannelOracle` 三个方法各自重复检查参数；`probes.run_probe` 对已冻结的源码再次做语法树检查；posthoc 重复运行 `check_source`、`series_matches_task` 和链路去重；断言源码文本的测试（`test_frontend_realtime_architecture.py`、`test_prompt_document_hygiene.py`、`test_api_contract.py` 的反向断言、`test_headroom_feedback.py:244`）。

过度设计和兼容代码：`/api/live`、`/v1/knowledge/query` 重复路由、MCP 本地后备、`build_reference_features` 空壳方法、旧字段迁移、`translate.py` 的 Claude SDK 转换代码、`nextStep → legacy` 前后端路径、`LEGACY_ALIASES`（用户已裁决删除）、从未改变的参数（`max_iterations`、`max_steps`、`FittedCorrector.label`、`_usage_dict` 的四处查找、`session_factory` 的九个只判空的回调）、`training/lightning.py` 及其副本和无引用的 `openai` 依赖、`scripts/` 下两个数字写死且标签写反的一次性脚本。

## 第四节：重复实现

experiments："拟合、预测、组装 segment"写了六遍；修正器预测、DM 检验、认证、`_aspect_lcb`、m0 岭回归各有两份；posthoc 把 `evaluate.py` 加载成两个模块对象。headroom：`_internal_split` 与 `_fit_ridge` 的切分重复；步长 16 写死两处。后端：LLM 配置清理三份（默认 `authMode` 各不相同）、`_bounded_int` 三份、关闭运行器的函数三份、三个重置函数各自维护删除列表。前端：图谱 LLM 配置两套表单且 payload 不一致；暂停按钮可用性由前端自行计算。测试：`_orchestrator` 辅助函数复制四份。

## 第五节：修改痕迹（违反 2.2）

后端：`orchestrator.py:389,703,1943`、`workspace_store.py:337`、`mcp/server.py:7`、`translate.py:84`、`knowledge_prompts.py:28`（"第一版用 CSV 表"）。headroom：`candidate_worker.py:11`、`certify.py:12`、`synthetic.py:813`。experiments：`loop.py:26-36`（"Design notes from the M2 review"）、`loop.py:1291`、`posthoc.py:234`（"in-run sha check"）、`probes.py`/`stage0.py` 多处。测试：以 "Regression for…" 开头的说明和 `legacy_*` 命名。过时的文档字符串：`feedback.py` 写"四个实验组"实际五个；`posthoc.py` 写"三组对比"实际五组。模块说明写在 `from __future__` 之后导致 `__doc__` 为空的文件：`loop.py`、`posthoc.py`、`batch.py`、`evaluate.py`、`calibrations.py`、`baselines.py`、`realdata.py`、`stage0.py`、headroom 下八个模块。

## 第六节：项目范围（Bitter Lesson）

论文实验只用到 `headroom/`、`experiments/`、`SdkRunner`、`settings/llm.py` 和 `MessageLog`，约 1.14 万行。交互系统（orchestrator、知识图谱、变体消融、MCP 桥接、前端）约 1.45 万行，论文实验完全不经过这部分代码。

- 🔴 核心：残差引导的候选演化循环（`experiments/loop.py`）加上认证的度量方法（residual → corrector → certify）。
- 🟡 待定：HarnessingTS 交互系统整体。投稿前不再投入精力，第二节的问题等需要使用界面时再修。
- ⚪ 装饰：`training/lightning.py` 及其副本、`scripts/` 下两个脚本、各类兼容接口、断言源码文本的测试。直接删除。
