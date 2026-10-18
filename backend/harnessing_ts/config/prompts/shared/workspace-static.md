Workspace path: {workspace_path}
Locale: {locale}
文件系统和 JSONL 日志是真相源；SDK session 是短命执行载体。
优先写入约定工件路径。`problem-contract` node 可以在 `data/raw/**` 中创建任务所需且尚不存在的原始数据，但不得覆盖或改写已有原始文件；任何解压后的派生副本、清洗、转换和特征结果必须写入 `data/processed/**` 或约定工件路径。其他 node 将 `data/raw/**` 视为严格只读。
该 workspace 是独立 uv Python 项目，包含自己的 `pyproject.toml`、`uv.lock` 和 `.venv/`。
运行任何 shell/python 命令都必须在 workspace 根目录使用 `uv run`，例如 `uv run python my_script.py`。
安装新的 Python 依赖必须使用 workspace 内的 uv 项目命令，例如 `uv add package-name` 或临时命令 `uv run --with package-name python script.py`；不要安装到系统 Python 或 HarnessingTS 源码项目环境。
读取 .docx reference 时直接读取系统生成的同名 `<docx>.txt` 文本副本；该副本在上传 reference 和初始化 workspace 布局时自动生成，无需自行抽取。
`knowledge_base/tables/*.csv`、`knowledge_base/indexes/**` 和 `knowledge_base/cache/**` 是 knowledge graph builder/reasoner 的内部存储。普通领域知识查询必须使用 MCP `mcp__ts_harness__query_knowledge`（若当前变体注入该工具），不要直接读取这些内部文件；只有用户明确要求调试知识库文件、CSV schema 或图谱构建错误时才可以直接读取。无知识图谱变体中，该 MCP 工具可能由 reference QA agent 直接读取 `references/**` 回答，也可能在 extractor-source 变体中只根据 `tools/reference-feature-extractor/**` 静态源码和规则回答；这不代表可以读取 `knowledge_base/**`。
调用 `mcp__ts_harness__query_knowledge` 时默认保持精简返回；只有用户明确要求原文证据、citations 或审计 trace 时，才设置 `includeEvidence=true`。
如果提供了 `mcp__ts_harness__inspect_reference_feature_extractor`，可用它读取由 `knowledge-to-tools` 节点生成并经后端校验通过的完整确定性源码、规则、reference 证据和 I/O 契约；不要自行改写其中的 reference 阈值或 judgment。
大型 JSON、inspect 结果和日志必须优先写入 workspace 工件，或使用分页、摘要和限定字段读取；不要把超出上下文承载能力的完整结果直接请求到对话中。
生成图片时优先同时保存可审计原图和适合报告/对话展示的缩略图，并记录图片尺寸、DPI 和文件路径。
