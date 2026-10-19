export type JsonMap = Record<string, any>;

export type FileTreeNode = {
  kind: 'dir' | 'file';
  name: string;
  path: string;
  size?: number;
  truncated?: boolean;
  childCount?: number;
  maxChildren?: number;
  maxDepth?: number;
  children?: FileTreeNode[];
};

export type Bootstrap = {
  variant?: JsonMap;
  state: JsonMap;
  timeline: JsonMap[];
  mainParts: JsonMap[];
  nodes: JsonMap[];
  nodePartsById: Record<string, JsonMap[]>;
  nodeSpecs: JsonMap[];
  fileTree: { root?: string; tree?: FileTreeNode; truncated?: boolean; entryCount?: number; maxDepth?: number; maxChildrenPerDir?: number } | null;
  llmConfig?: JsonMap;
  runtimeSettings?: JsonMap;
  knowledgeGraph?: JsonMap;
  knowledgeBaseSummary?: JsonMap;
  knowledgeGraphParts?: JsonMap[];
  chainSummary?: JsonMap;
  chainSummaryBuild?: JsonMap;
  chainSummaryParts?: JsonMap[];
  referenceFeatureBuild?: JsonMap;
  referenceFeatureTool?: JsonMap;
  dryRun?: boolean;
  debugEnabled?: boolean;
  runtime?: {
    running?: boolean;
    knowledgeGraphRunning?: boolean;
    chainSummaryRunning?: boolean;
    pipelineComplete?: boolean;
    workspaceUv?: JsonMap | null;
  };
  execution?: {
    actorId?: string;
    sessionId?: string | null;
    phase?: 'ready' | 'running' | 'pausing' | 'paused' | 'recovering' | 'failed';
    epoch?: number;
    operationId?: string | null;
    checkpointSeq?: number;
    canPause?: boolean;
    canForceStop?: boolean;
    canContinue?: boolean;
    lastError?: string | null;
  };
  knowledgeGraphBuild?: JsonMap;
  knowledgeGraphLlmConfig?: JsonMap;
};

// Experiment-run API (docs/experiments-api.md).
export type ProcessStatus = {
  state: 'running' | 'stopped' | 'none';
  pid: number | null;
  specFile: string | null;
  startedAt: string | null;
};

export type SpecSummary = {
  name: string;
  file: string;
  description: string;
  cellCount: number;
  tasks: string[];
  arms: string[];
  seeds: number[];
  budget: number;
};

export type RunSummary = {
  name: string;
  process: ProcessStatus;
  cellsDone: number;
  cellsTotal: number | null;
  lastActivityAt: string | null;
};

export type CellState = 'pending' | 'running' | 'done' | 'failed';

export type CellSummary = {
  cell: string;
  task: string;
  arm: string;
  seed: number;
  roundsDone: number;
  budget: number | null;
  accepted: number;
  incumbent: string;
  saturated: boolean | null;
  state: CellState;
  lastDiagMse: number | null;
  incumbentMse: number | null;
};

export type BatchEvent = {
  event: 'ok' | 'fail' | string;
  cell: string;
  exitCode: number | null;
  attempt: number;
  seconds: number | null;
  at: string;
  detail: string | null;
};

export type RunSnapshot = {
  name: string;
  process: ProcessStatus;
  spec: SpecSummary | null;
  cells: CellSummary[];
  batchLogTail: string;
  batchEvents: BatchEvent[];
};

export type TokenUsage = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  reasoningTokens: number;
};

export type RoundRow = {
  round: number;
  candidateId: string;
  parent: string;
  status: string;
  diagMse: number | null;
  dmP: number | null;
  accepted: boolean;
  error: string | null;
  evalNote: string | null;
  targetAspect: string | null;
  inspirationParent: string | null;
  judgeCause: string | null;
  judgeFailed: boolean;
  tokens: TokenUsage;
  judgeTokens: TokenUsage | null;
  seconds: number | null;
};

export type CertRow = {
  round: number;
  incumbentId: string;
  saturated: boolean;
  level: number;
  failures: Array<{ label: string; ucb: number; threshold: number }>;
};

export type CellDetail = {
  cell: string;
  task: string;
  arm: string;
  seed: number;
  state: CellState;
  budget: number | null;
  rounds: RoundRow[];
  certifications: CertRow[];
  incumbentCurve: IncumbentPoint[];
  logTail: string;
};

// Round 0 is the m0 baseline; mse is the current incumbent's diag MSE after that round.
export type IncumbentPoint = {
  round: number;
  mse: number | null;
  accepted: boolean;
};

export type PosthocCard = {
  label: string;
  aspect: string;
  delta: number;
  lcb: number;
  ucb: number;
};

export type RoundDetail = {
  round: number;
  proposal: Record<string, any>;
  posthoc: PosthocCard[] | null;
  parts: JsonMap[];
};

// posthoc analysis.json (experiments/posthoc.py), served verbatim by GET .../analysis.
export type EndpointStats = {
  n: number;
  median: number | null;
  q25: number | null;
  q75: number | null;
  iqr: number | null;
};

export type EndpointRow = {
  task: string;
  arm: string;
  nRuns: number;
  e1: EndpointStats;
  e2: EndpointStats;
};

export type Contrast = {
  contrast: string;
  nBlocks: number;
  statistic: number | null;
  pValue: number | null;
  permLow: number | null;
  permHigh: number | null;
  permutations: string | null;
};

export type Analysis = {
  runsDir?: string;
  runs?: Array<{ run?: string; e1Kind?: string; error?: string }>;
  incomplete?: string[];
  endpointTable?: Array<Partial<EndpointRow>>;
  contrasts?: Array<Partial<Contrast>>;
  audit?: { pathViolationRuns?: string[]; gainRatioFlagRuns?: string[] };
};

export type AnalysisResponse = {
  process: { state: 'running' | 'stopped' | 'none'; pid: number | null };
  analysis: Analysis | null;
  logTail: string;
};
