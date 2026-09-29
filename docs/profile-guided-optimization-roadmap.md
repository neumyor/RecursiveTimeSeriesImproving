# 通用时序模型的 Profile 驱动优化开发目标

## 1. 当前共识

项目的二次开发目标是：在指定硬件、运行占用率和性能约束下，通过可复现的 profile 工具获取运行证据，在保持预测准确度的前提下优化通用时序模型的运行效率。

新的核心问题不是从领域资料中提取 ECG 特征，而是建立以下证据链：

```text
模型候选
→ 目标硬件上的 benchmark
→ profile hotspot
→ 结构化 evidence
→ 模型修改
→ 准确度与资源约束判定
→ 接受或淘汰候选
```

`knowledge-to-tools`、知识图谱、参考特征提取器和 ECG 特定的领域知识检索不属于新的核心闭环。它们可以保留在旧实现的 Git 历史中，作为未来可选的领域插件，但不应出现在新的优化 workspace、工具池、节点 prompt 或默认前端流程中。

新的系统应优先支持通用时序模型，包括分类、回归、预测和异常检测任务；模型框架与硬件通过 adapter 接入，不把 ECG 特征、文件格式或某一种模型结构写入核心协议。

## 2. 目标架构

新的节点链为：

```text
problem-contract
→ profile-bootstrap
→ profile-guided-optimization（可重复）
→ final-summary
```

`profile-bootstrap` 是一个节点，但它不是单纯的 baseline 节点。它负责读取任务契约、准备并验证 profile 工具链、解决目标硬件上的依赖和兼容性问题，然后生成一次不可变 baseline。实际的硬件探测、benchmark 和 profile 测量由后端确定性执行，不能交给 LLM 自由编写和解释。

`profile-guided-optimization` 只使用已经通过 bootstrap 验证的工具，负责提出候选模型、执行候选 benchmark、查询 profile evidence、修改模型并进行约束判定。后续迭代不再重复安装 profiler、解决设备兼容性或重新设计采集脚本。

`profile-bootstrap` 的内部流程：

```text
读取 optimization contract
→ 探测并锁定硬件指纹
→ 选择或生成受约束的 profiler adapter
→ 准备一次性依赖环境
→ 运行 profile 工具自检
→ 运行 baseline accuracy/performance benchmark
→ 生成并冻结 tool manifest 与 baseline manifest
```

## 3. 删除和保留范围

新的优化模式中删除以下核心依赖：

- `knowledge-to-tools`
- `KGR`、`RQA`、`FQA`
- `query_knowledge`
- `reference-feature-extractor`
- `knowledge_base`、`knowledge_graph`
- 依赖领域资料的 case attribution
- 将 ECG 特征作为通用模型优化依据的 prompt 和 UI

当前相关实现主要位于 `backend/harnessing_ts/knowledge_graph.py`、`knowledge_prompts.py`、`reference_feature_extractor.py`，以及 `variants/profiles.py`、`tools/compose_tools.py` 和相应的节点配置。

现有 case review 不原样保留。它应拆成通用的：

- `error-slice review`：分析准确度下降集中在哪些数据切片、输入形态或标签组。
- `performance-outlier review`：分析延迟、内存或利用率异常对应的输入、算子和 batch。

`ADA` 可以保留，用于自适应停止；`SUB` 只作为可选能力，默认关闭，避免并行 profile 造成硬件竞争和测量噪声。

## 4. 优化模式和配置

第一版建议使用单独的 `PFO`（Profile-guided Optimization）能力，而不是继续扩展知识型 feature-code 组合。推荐模式：

```text
NOD-PFO-ADA
```

可选并行版本：

```text
NOD-PFO-SUB-ADA
```

若后续确认不再需要 ablation 组合，可将 variant 简化为普通配置：

```json
{
  "optimizationMode": "profile-guided",
  "parallelCandidates": false,
  "adaptiveStop": true
}
```

工作区配置文件为 `config.optimization.json`，至少包含：

```json
{
  "schemaVersion": 1,
  "hardware": {
    "backend": "mps",
    "device": "mps",
    "batchSizes": [1, 8],
    "concurrency": 1
  },
  "accuracy": {
    "metric": "macro_f1",
    "direction": "higher",
    "minimum": 0.91,
    "maxDropFromBaseline": 0.005,
    "confidenceLevel": 0.95
  },
  "performance": {
    "p95LatencyMs": 30,
    "maxPeakMemoryMb": 2048,
    "maxCpuUtilPct": 85,
    "maxGpuUtilPct": null,
    "minThroughputPerSecond": 20
  },
  "profile": {
    "warmupRuns": 10,
    "measurementRuns": 50,
    "traceRuns": 3,
    "timeoutSeconds": 300
  }
}
```

硬约束、软目标和不可测指标必须明确区分。缺少必需测量时，候选应判定为 `unevaluable`，不能默认通过。

## 5. 标准模型和数据接口

候选模型不应依赖 ECG 特定脚本。每个模型需要提供统一 adapter：

```python
class ModelAdapter:
    def load(self, config): ...
    def predict_batch(self, model, batch): ...
    def evaluate(self, model, dataset, split): ...
    def profile_config(self): ...
```

建议的候选目录：

```text
models/candidates/<candidate-id>/
├── model.py
├── predict.py
├── evaluate.py
├── metadata.json
└── README.md
```

统一接口必须固定预处理、数据切分、输出语义和评测指标，防止候选通过改变数据处理方式获得虚假的性能提升。

## 6. Profile 工具链

建议新增：

```text
backend/harnessing_ts/profiling/
├── schema.py
├── hardware.py
├── runner.py
├── benchmark.py
├── collectors.py
├── constraints.py
├── evidence.py
└── artifacts.py
```

硬件 adapter 第一版支持：

- CPU：wall-clock、RSS、CPU utilization。
- NVIDIA CUDA：`torch.profiler`、NVML、CUDA synchronize。
- Apple MPS：CPU profile、MPS synchronize、MPS memory、host RSS。
- 无加速器环境：CPU fallback。

当前开发环境是 Apple Silicon/MPS。MPS GPU utilization 未必能够像 NVML 一样可靠取得，因此必须显式记录 `unsupported`，不能用 0 代替。

profile runner 必须提供固定 warmup、重复次数、timeout、同步和输出格式；不得允许后续节点随意修改采集逻辑。可使用 `cProfile`、`torch.profiler` 和硬件专用采集器，但所有结果必须带有硬件、软件、模型、数据和 profile 配置指纹。

## 7. Evidence 和运行产物

建议的 workspace 布局：

```text
user/
  problem-contract.md
  optimization-contract.md
  iteration-state.md
  final-solution.md

models/
  baseline/
  candidates/

benchmarks/
  baseline/
  candidates/

profiles/
  baseline/
  candidates/

evidence/
  index.jsonl
  hotspot/
  comparisons/

runs/iterations/
reports/iterations/
reports/final-summary.md

state/hardware.json
state/profile-bootstrap.json
state/optimization.json
state/constraints.json
```

每条 profile evidence 至少包含：

```json
{
  "evidenceId": "E-PROFILE-00017",
  "candidateId": "candidate-int8",
  "profileRunId": "profile-003",
  "kind": "hotspot",
  "operator": "aten::conv1d",
  "location": "models/candidates/candidate-int8/model.py:87",
  "metric": "p95_latency_ms",
  "baselineValue": 42.1,
  "candidateValue": 28.4,
  "deltaPct": -32.5,
  "unit": "ms",
  "sourceArtifact": "profiles/candidates/candidate-int8/trace.json",
  "reproducible": true
}
```

所有优化结论必须能追溯：

```text
profileRunId
→ evidenceId
→ hotspot
→ model change
→ benchmarkRunId
→ constraintDecision
```

## 8. 分阶段开发目标

### 阶段一：清理核心边界，建立优化契约

目标：让新模式不再依赖知识图谱、KTL 和 ECG 特征。

主要工作：

- 在 `variants/profiles.py` 中增加 `PFO` 能力，取消新模式对 KGR/RQA/FQA 的强制依赖。
- 从新模式工具池移除 `query_knowledge` 和 reference feature tools。
- 新增 `config.optimization.json` 的 schema、读取和校验。
- 新增通用 `ModelAdapter`、`DatasetAdapter` 和 benchmark report 类型。
- 更新 `schema.py`、`workspace_layout.py`、`workspace_store.py` 的基础结构。
- 新增 `problem-contract` 对硬件、准确度和性能约束的要求。

验收标准：

- `NOD-PFO-ADA` 可以启动并通过变体校验。
- 旧知识型变体不会影响新模式。
- 没有 references 或 knowledge base 时，新模式仍能初始化。
- 配置错误会 fail fast，并给出缺失字段和单位。

### 阶段二：实现 profile-bootstrap

目标：在目标硬件上一次性准备并验证 profile 工具链。

主要工作：

- 新增 `profile-bootstrap` 节点规范、prompt 和输出路径。
- 实现 CPU、CUDA、MPS hardware adapter。
- 实现 profile runner、benchmark runner 和硬件指纹。
- 生成并校验 `tools/profiling/manifest.json`。
- 实现依赖安装、导入检查、模型 smoke test、profile smoke test。
- 运行 baseline accuracy、latency、throughput、memory 和可用 utilization 指标。
- 写入 `state/profile-bootstrap.json` 和 baseline manifests。

验收标准：

- 没有完成 bootstrap，不能进入优化节点。
- profile runner 能在同一模型上重复运行。
- profile 失败、依赖缺失和指标不可测会明确失败。
- 生成的工具版本、模型、数据和硬件都有 fingerprint。
- profile 工具依赖只在 bootstrap 阶段安装一次。

### 阶段三：实现 profile evidence 和约束判定

目标：把原始 profile 转化为可信、可查询、可审计的证据。

主要工作：

- 定义 `ProfileReport`、`BenchmarkReport`、`EvidenceRecord`、`ConstraintDecision`。
- 提取 operator hotspot、内存热点、延迟分布和利用率摘要。
- 实现 baseline 与候选的对比。
- 实现准确度 gate、延迟 gate、内存 gate、利用率 gate 和不可测判定。
- 实现 Pareto frontier 和确定性候选排序。
- 增加 profile stale 检测。

验收标准：

- 缺少 profile evidence 的候选不能被标记为优化成功。
- 准确度低于下限的候选永远不能被接受。
- 指纹变化会使旧 evidence 失效。
- 同一输入下可以从 candidate 追溯到 profile、evidence 和约束结果。

### 阶段四：接入 profile-guided-optimization 节点

目标：让 agent 根据已经验证的 profile 工具进行模型优化。

主要工作：

- 新增 `profile-guided-optimization` 节点，或将现有 `iterative-solving` 重构为该节点。
- 每轮先生成候选，再调用统一 benchmark/profile API。
- 禁止节点安装新的 profiler 或修改 bootstrap 工具。
- 将 error-slice review 和 performance-outlier review 加入迭代报告。
- 扩展 `record_run`，强制保存 profile、benchmark 和 constraint metadata。
- 只允许满足 gate 的候选更新 `tools/registry.json` 或最终模型 registry。

验收标准：

- agent 能从 hotspot evidence 提出可审计的模型修改。
- 每轮候选都使用同一套 profile 工具和 baseline。
- 候选失败不会污染当前 accepted model。
- `loopDecision` 仍能正确控制继续迭代或进入 final-summary。

### 阶段五：增加 API、SSE 和前端工作台

目标：让用户能配置约束、查看 profile、比较候选并审查 evidence。

主要工作：

- 增加 baseline、profile run、evidence、constraint 和 Pareto API。
- 将 profile 状态加入 bootstrap/live payload。
- 增加 `profile_started`、`profile_evidence_ready`、`benchmark_completed` 和 `constraint_decision` SSE 事件。
- 删除新模式中的知识图谱和 reference feature 页面。
- 增加硬件配置、baseline 面板、hotspot 表、候选比较和性能图表。

验收标准：

- UI 能显示当前硬件和 baseline。
- UI 能展示每个候选的准确度、延迟、内存和约束结果。
- evidence 能打开原始 profile artifact。
- UI 不把不可测指标显示成 0 或通过。

### 阶段六：稳定性和真实模型验证

目标：确认系统能在真实通用时序模型上稳定运行。

主要工作：

- 使用合成时序模型完成端到端测试。
- 使用至少一个真实时序分类或预测模型验证。
- 验证 CPU、CUDA 或 MPS 中至少两个硬件适配路径。
- 验证 profile 噪声、超时、进程回收和 workspace 隔离。
- 更新 README、AGENT.md、architecture 文档和测试说明。

验收标准：

- 原有 DSH session recovery、pause/resume 和 MCP capability 测试不回归。
- profile runner 不会遗留无法管理的进程。
- 旧变体仍能按兼容策略运行，或者有明确迁移错误提示。
- final summary 能完整呈现准确度保持和效率提升的证据链。

## 9. 重点代码改造位置

| 文件 | 目标 |
|---|---|
| `variants/profiles.py` | 增加 PFO，移除新模式的知识能力依赖 |
| `tools/compose_tools.py` | 注入 profile、benchmark、evidence 和 constraint 工具 |
| `mcp/server.py` | 增加 profile MCP schema 和权限控制 |
| `orchestrator.py` | 管理 bootstrap、profile runner 和候选生命周期 |
| `node_state.py` | 加入 bootstrap 前置条件和新节点路由 |
| `schema.py` | 增加结构化性能、profile 和约束类型 |
| `workspace_store.py` | 持久化 baseline、profile、evidence 和决策 |
| `workspace_layout.py` | 创建新的 models、benchmarks、profiles、evidence 布局 |
| `api/payloads.py` | 聚合优化工作台所需的状态和产物 |
| `server.py` | 增加 profile、benchmark 和 constraint API |
| `frontend/src/main.ts` | 删除知识图谱主流程，增加 profile 工作台 |
| `backend/harnessing_ts/config/nodes/` | 新增 bootstrap 和 profile-guided-optimization 规范 |

## 10. 待实现前需要确认的信息

正式实现前需要确定：

1. 目标硬件型号和操作系统。
2. “占用率”具体指 CPU、GPU、显存、主存、功耗，还是组合约束。
3. 第一批支持的模型框架。
4. baseline 模型入口和预测命令。
5. 数据集、固定切分和准确度指标。
6. 允许的准确度下降范围。
7. 延迟、吞吐量、内存和利用率阈值。
8. 是否允许量化、剪枝、蒸馏、混合精度、编译和 batch 调整。
9. 是否允许安装系统级 profile 工具。
10. 最终模型交付形式，例如 Python、TorchScript、ONNX 或 TensorRT。

