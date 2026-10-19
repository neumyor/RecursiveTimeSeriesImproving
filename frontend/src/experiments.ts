// Experiment-run pages: start a batch from a spec, watch cells and rounds, read the posthoc analysis.
// Backed by the routes in docs/experiments-api.md. One poll timer (5 s) runs only while the view
// is open; all rendering is plain HTML strings plus click delegation on the root.
import { fetchJson, postJson } from './api';
import type {
  Analysis,
  AnalysisResponse,
  BatchEvent,
  CellDetail,
  CellState,
  CellSummary,
  CertRow,
  JsonMap,
  ProcessStatus,
  RoundDetail,
  RoundRow,
  RunSnapshot,
  RunSummary,
  SpecSummary,
} from './types';

const API = '/api/experiments';
const POLL_MS = 5000;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/; // same rule as the server's fullmatch

// Plain-language arm names for the four arms the specs use; unknown arms show as-is.
const ARM_INFO: Record<string, { label: string; note: string }> = {
  scalar: { label: '标量反馈', note: '每轮只给误差指标' },
  fixed: { label: '固定缺陷卡', note: '每轮附一份预置的缺陷清单' },
  'self-built': { label: '自建探针', note: '智能体自己编写诊断探针' },
  augmented: { label: '缺陷卡 + 自建探针', note: '预置缺陷清单，外加智能体自建探针' },
  placebo: { label: '占位对照', note: '打乱后的占位反馈，用于对照' },
};
const armLabel = (arm: string) => ARM_INFO[arm]?.label ?? arm;
const armNote = (arm: string) => ARM_INFO[arm]?.note ?? '';
const seedList = (seeds: number[]) => seeds.join('、');

export type ExperimentsDeps = {
  root: HTMLElement;
  escapeHtml: (value: any) => string;
  hydrateIcons: (root: ParentNode) => void;
  renderStable: (key: string, value: any, renderFn: () => void) => void;
  runAction: (fn: () => Promise<void>) => Promise<void>;
  emptyState: (text: string, icon?: string) => string;
  messageHtml: (part: JsonMap) => string;
  normalizeChatParts: (parts: JsonMap[]) => JsonMap[];
  bindToolCards: (root: HTMLElement) => void;
  goSettings: () => void;
};

type Route = { run: string | null; cell: string | null; round: number | null };

const safeDecode = (part: string) => {
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
};
const errText = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function createExperiments(deps: ExperimentsDeps) {
  const { root, escapeHtml: esc, hydrateIcons, renderStable, emptyState } = deps;

  const s = {
    active: false,
    shellBuilt: false,
    route: { run: null, cell: null, round: null } as Route,
    pageKey: '',
    gen: 0, // bumped whenever the page skeleton is rebuilt, so every card redraws once
    specs: null as SpecSummary[] | null,
    specsError: '',
    runs: null as RunSummary[] | null,
    runsError: '',
    run: null as RunSnapshot | null,
    cell: null as CellDetail | null,
    cellRun: '',
    round: null as RoundDetail | null,
    roundKey: '',
    analysis: null as AnalysisResponse | null,
    analysisRun: '',
    analysisError: '',
    analysisBusy: false,
    analysisForce: false,
    error: '',
    updatedAt: '',
    timer: null as number | null,
    flight: null as Promise<void> | null,
    starting: false,
    startError: '',
    form: { spec: '', name: '', workers: 4, only: '', nameTouched: false }, // spec = SpecSummary.file
  };

  // ------------------------------------------------------------ routing

  function pathFor(route: Route): string {
    const parts = ['/experiments'];
    if (route.run) parts.push(encodeURIComponent(route.run));
    if (route.run && route.cell) parts.push(encodeURIComponent(route.cell));
    if (route.run && route.cell && route.round !== null) parts.push(String(route.round));
    return parts.join('/');
  }

  function parseRoute(): Route {
    const parts = window.location.pathname.split('/').filter(Boolean).slice(1).map(safeDecode);
    const run = parts[0] || null;
    const cell = run ? parts[1] || null : null;
    const round = cell && parts[2] !== undefined && /^\d+$/.test(parts[2]) ? Number(parts[2]) : null;
    return { run, cell, round };
  }

  function navigate(path: string) {
    if (path !== window.location.pathname) history.pushState({}, '', path);
    syncFromLocation();
  }

  function syncFromLocation() {
    s.route = parseRoute();
    s.error = '';
    if (s.run && s.run.name !== s.route.run) s.run = null;
    if (s.cell && (s.cellRun !== s.route.run || s.cell.cell !== s.route.cell)) s.cell = null;
    const roundKey = `${s.route.run}/${s.route.cell}/${s.route.round}`;
    if (s.round && s.roundKey !== roundKey) s.round = null;
    if (s.analysisRun !== (s.route.run ?? '')) {
      s.analysis = null;
      s.analysisError = '';
      s.analysisRun = s.route.run ?? '';
    }
    renderPage();
    requestTick();
  }

  // ------------------------------------------------------------ polling

  function setActive(active: boolean) {
    if (active === s.active) return;
    s.active = active;
    if (active) {
      buildShell();
      s.timer = window.setInterval(() => void tick(false), POLL_MS);
      syncFromLocation();
    } else {
      if (s.timer !== null) window.clearInterval(s.timer);
      s.timer = null;
    }
  }

  // Run a poll now; if one is already in flight, run right after it ends (the route may have changed).
  function requestTick() {
    if (!s.active) return;
    void (s.flight ?? Promise.resolve()).then(() => tick(true));
  }

  // Timer ticks skip while the tab is hidden; route changes and manual refreshes always load.
  async function tick(force: boolean) {
    if (!s.active || s.flight || (document.hidden && !force)) return;
    const { run, cell, round } = s.route;
    const key = pathFor(s.route);
    const work = (async () => {
      let error = '';
      try {
        error = !run ? await loadList(key) : !cell ? await loadRun(run, key) : await loadCell(run, cell, round, key);
      } catch (caught) {
        error = errText(caught);
      }
      if (!stillHere(key)) return;
      s.error = error;
      if (!error) s.updatedAt = new Date().toLocaleTimeString();
      renderPage();
    })();
    s.flight = work.finally(() => {
      s.flight = null;
    });
    await s.flight;
  }

  function stillHere(key: string) {
    return s.active && pathFor(s.route) === key;
  }

  // Each loader returns the error text to show in the status line ('' when everything loaded).
  async function loadList(key: string): Promise<string> {
    const [runs, specs] = await Promise.allSettled([
      fetchJson<RunSummary[]>(`${API}/runs`),
      s.specs ? Promise.resolve(s.specs) : fetchJson<SpecSummary[]>(`${API}/specs`),
    ]);
    if (!stillHere(key)) return '';
    if (runs.status === 'fulfilled') {
      s.runs = runs.value;
      s.runsError = '';
    } else {
      s.runsError = errText(runs.reason);
    }
    if (specs.status === 'fulfilled') {
      s.specs = specs.value;
      s.specsError = '';
    } else {
      s.specsError = errText(specs.reason);
    }
    return [s.runsError && `运行列表：${s.runsError}`, s.specsError && `规格：${s.specsError}`].filter(Boolean).join('；');
  }

  async function loadRun(name: string, key: string): Promise<string> {
    const base = `${API}/runs/${encodeURIComponent(name)}`;
    // The analysis is only re-read while it is unknown, being generated, or just requested.
    const wantAnalysis = !s.analysis || s.analysisForce || s.analysis.process.state === 'running';
    const analysisReq = wantAnalysis
      ? fetchJson<AnalysisResponse>(`${base}/analysis`).then((value) => ({ value }), (reason) => ({ reason }))
      : null;
    const run = await fetchJson<RunSnapshot>(base);
    const analysis = analysisReq && (await analysisReq);
    if (!stillHere(key)) return '';
    s.run = run;
    if (analysis) {
      s.analysisForce = false;
      if ('value' in analysis) {
        s.analysis = analysis.value;
        s.analysisError = '';
      } else {
        s.analysisError = errText(analysis.reason);
      }
    }
    return '';
  }

  async function loadCell(run: string, cell: string, round: number | null, key: string): Promise<string> {
    const base = `${API}/runs/${encodeURIComponent(run)}/cells/${encodeURIComponent(cell)}`;
    // Round detail is static: fetched once per route change. The run snapshot only feeds the
    // cell page's "run this cell alone" button, so a failure there is not an error.
    const [detail, roundDetail, snapshot] = await Promise.all([
      round === null || !s.cell ? fetchJson<CellDetail>(base) : s.cell,
      round !== null && !s.round ? fetchJson<RoundDetail>(`${base}/rounds/${round}`) : null,
      round === null ? fetchJson<RunSnapshot>(`${API}/runs/${encodeURIComponent(run)}`).catch(() => s.run) : null,
    ]);
    if (!stillHere(key)) return '';
    s.cell = detail;
    s.cellRun = run;
    if (roundDetail) {
      s.round = roundDetail;
      s.roundKey = `${run}/${cell}/${round}`;
    }
    if (snapshot) s.run = snapshot;
    return '';
  }

  async function refreshNow() {
    if (s.flight) await s.flight;
    await tick(true);
  }

  // ------------------------------------------------------------ shell and skeletons

  function buildShell() {
    if (s.shellBuilt) return;
    s.shellBuilt = true;
    root.innerHTML = `
      <header class="chain-header exp-header">
        <div class="chain-title-block">
          <div class="eyebrow"><span data-icon="FlaskConical"></span><span>实验</span></div>
          <h2>实验运行</h2>
          <p class="exp-header-lead">启动实验、查看进度、检查每一轮的记录。一个单元格 = 任务 × 反馈臂 × 种子。</p>
          <nav id="expCrumbs" class="exp-crumbs" aria-label="实验导航"></nav>
        </div>
        <div class="chain-header-actions">
          <span id="expStatus" class="exp-status"></span>
          <button type="button" class="chain-back-btn" data-exp-action="settings"><span data-icon="Settings2"></span><span>设置</span></button>
        </div>
      </header>
      <div class="exp-scroll"><div id="expPage" class="exp-page"></div></div>
    `;
    hydrateIcons(root);
  }

  const el = (id: string) => root.querySelector<HTMLElement>(`#${id}`);

  function pageKind(): 'list' | 'run' | 'cell' | 'round' {
    const { run, cell, round } = s.route;
    return !run ? 'list' : !cell ? 'run' : round === null ? 'cell' : 'round';
  }

  function buildSkeleton(kind: string): string {
    if (kind === 'list') {
      return `<section class="exp-guide">
          <div class="exp-guide-step"><span class="exp-guide-num">1</span><div><strong>选实验方案</strong><p>方案决定任务、反馈臂、种子和每格的轮数。</p></div></div>
          <div class="exp-guide-step"><span class="exp-guide-num">2</span><div><strong>启动运行</strong><p>填运行名称、选并行数，点“启动运行”。</p></div></div>
          <div class="exp-guide-step"><span class="exp-guide-num">3</span><div><strong>查看进度</strong><p>点矩阵里的方格进入单元格，看曲线和逐轮记录。</p></div></div>
        </section>
        <div class="exp-list-grid"><section id="expStart" class="exp-card"></section><section id="expRuns" class="exp-card"></section></div>`;
    }
    const log = (title: string) => `
      <details class="exp-card exp-log">
        <summary class="panel-heading"><span data-icon="TerminalSquare"></span><span>${title}</span></summary>
        <div id="expLogBody"></div>
      </details>`;
    if (kind === 'run') {
      return `<section id="expRunHead" class="exp-card"></section>
        <section id="expMatrix" class="exp-card"></section>
        <section id="expAnalysis" class="exp-card"></section>
        ${log('batch 日志与最近事件')}`;
    }
    if (kind === 'cell') {
      return `<section id="expCellHead" class="exp-card"></section>
        <section id="expChart" class="exp-card"></section>
        <section id="expRounds" class="exp-card"></section>
        <section id="expCerts" class="exp-card"></section>
        ${log('单元格日志尾部')}`;
    }
    return `<section id="expRoundHead" class="exp-card"></section>
      <section id="expProposal" class="exp-card"></section>
      <section id="expPosthoc" class="exp-card"></section>
      <section id="expSession" class="exp-card"></section>`;
  }

  // Redraw a block while keeping what the operator set in the old DOM: each log's scroll
  // position (pinned to the bottom if it was there) and which <details class="exp-keep"> are open.
  function keepUi(target: HTMLElement, redraw: () => void) {
    const logs = [...target.querySelectorAll<HTMLElement>('.exp-pre')].map((pre) => ({
      top: pre.scrollTop,
      bottom: pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4,
    }));
    const opens = [...target.querySelectorAll<HTMLDetailsElement>('details.exp-keep')].map((details) => details.open);
    redraw();
    target.querySelectorAll<HTMLDetailsElement>('details.exp-keep').forEach((details, index) => {
      details.open = Boolean(opens[index]);
    });
    target.querySelectorAll<HTMLElement>('.exp-pre').forEach((pre, index) => {
      const before = logs[index];
      pre.scrollTop = !before || before.bottom ? pre.scrollHeight : before.top;
    });
  }

  function renderPage() {
    if (!s.active) return;
    const page = el('expPage');
    if (!page) return;
    const kind = pageKind();
    const key = pathFor(s.route);
    if (s.pageKey !== key) {
      s.pageKey = key;
      s.gen += 1;
      page.innerHTML = buildSkeleton(kind);
      hydrateIcons(page);
    }
    // Fixed keys; the skeleton generation rides inside the value so a rebuilt page redraws once.
    const stable = (name: string, value: any, fn: (target: HTMLElement) => void, error = s.error) => {
      const target = el(name);
      if (!target) return;
      renderStable(`exp-${name}`, { gen: s.gen, value, error }, () => {
        keepUi(target, () => fn(target));
        hydrateIcons(target);
      });
    };
    renderStable('exp-crumbs', s.route, renderCrumbs);
    renderStatus();
    renderRail();

    if (kind === 'list') {
      stable('expStart', s.specs, renderStart, s.specsError);
      stable('expRuns', s.runs, renderRuns, s.runsError);
      updateStartHint();
    } else if (kind === 'run') {
      const run = s.run;
      stable('expRunHead', run && { name: run.name, process: run.process, spec: run.spec, states: run.cells.map((c) => c.state) }, renderRunHead);
      stable('expMatrix', run && { spec: run.spec, cells: run.cells }, renderMatrix);
      stable('expAnalysis', { analysis: s.analysis, busy: s.analysisBusy }, renderAnalysis, s.analysisError);
      stable('expLogBody', run && { log: run.batchLogTail, events: run.batchEvents }, renderRunLog);
    } else if (kind === 'cell') {
      const cell = s.cell;
      const process = s.run && s.run.name === s.route.run ? s.run.process : null;
      stable('expCellHead', cell && { cell, process }, renderCellHead);
      stable('expChart', cell && [cell.incumbentCurve, cell.certifications, cell.budget, cell.rounds.map((r) => r.candidateId)], renderChart);
      stable('expRounds', cell && cell.rounds, renderRounds);
      stable('expCerts', cell && cell.certifications, renderCerts);
      stable('expLogBody', cell && cell.logTail, renderCellLog);
    } else {
      stable('expRoundHead', { row: roundRow(), cell: s.cell && s.cell.rounds.map((r) => r.round), n: s.route.round }, renderRoundHead);
      stable('expProposal', s.round && s.round.proposal, renderProposal);
      stable('expPosthoc', s.round && { posthoc: s.round.posthoc }, renderPosthoc);
      stable('expSession', s.round && s.round.parts, renderSession);
    }
  }

  function renderCrumbs() {
    const nav = el('expCrumbs');
    if (!nav) return;
    const { run, cell, round } = s.route;
    if (!run) {
      nav.innerHTML = '';
      return;
    }
    const items: Array<{ label: string; route: Route | null }> = [{ label: '所有运行', route: { run: null, cell: null, round: null } }];
    items.push({ label: run, route: { run, cell: null, round: null } });
    if (run && cell) items.push({ label: cell, route: { run, cell, round: null } });
    if (run && cell && round !== null) items.push({ label: `第 ${round} 轮`, route: null });
    items[items.length - 1].route = null;
    nav.innerHTML = items
      .map((item, index) => {
        const sep = index ? '<span class="exp-crumb-sep">/</span>' : '';
        return item.route
          ? `${sep}<button type="button" class="exp-crumb" data-exp-go="${esc(pathFor(item.route))}">${esc(item.label)}</button>`
          : `${sep}<span class="exp-crumb current">${esc(item.label)}</span>`;
      })
      .join('');
  }

  function renderRail() {
    const rail = document.getElementById('expRailRun');
    if (!rail) return;
    const run = s.route.run ? s.run : null;
    const runs = s.runs;
    renderStable('exp-rail', { route: s.route, run: run && [run.name, run.process, run.cells.map((c) => c.state)], runs }, () => {
      if (!s.route.run) {
        const running = runs?.filter((item) => item.process.state === 'running').length ?? 0;
        rail.innerHTML = `<div class="meta">${runs ? `共 ${esc(runs.length)} 个运行，${esc(running)} 个运行中。` : '加载中…'}</div>`;
      } else if (!run) {
        rail.innerHTML = `<div class="meta">${esc(s.route.run)}</div>`;
      } else {
        const totals = runTotals(run);
        rail.innerHTML = `<dl class="kv exp-kv">
          <dt>运行</dt><dd>${esc(run.name)}</dd>
          <dt>进程</dt><dd>${processChip(run.process)}</dd>
          <dt>方案</dt><dd>${esc(run.spec?.name || run.process.specFile || '—')}</dd>
          <dt>进度</dt><dd>${esc(totals.done)}/${esc(totals.total)} 格${totals.failed ? ` · ${esc(totals.failed)} 失败` : ''}</dd>
        </dl>${progressBar(totals.done, totals.total)}`;
      }
      hydrateIcons(rail);
    });
  }

  function renderStatus() {
    const status = el('expStatus');
    if (!status) return;
    status.classList.toggle('error', Boolean(s.error));
    status.textContent = s.error
      ? `刷新失败：${s.error}`
      : s.updatedAt ? `每 5 秒自动刷新 · 更新于 ${s.updatedAt}` : '加载中…';
  }

  // ------------------------------------------------------------ formatting helpers

  const isNum = (value: any): value is number => typeof value === 'number' && Number.isFinite(value);
  const fmtMse = (value: number | null | undefined) => (isNum(value) ? String(Number(value.toPrecision(4))) : '—');
  const fmtP = (value: number | null | undefined) => (isNum(value) ? (value < 0.001 ? '<0.001' : value.toFixed(3)) : '—');
  const fmtSigned = (value: number | null | undefined) => (isNum(value) ? `${value > 0 ? '+' : ''}${Number(value.toPrecision(3))}` : '—');
  const fmtK = (value: number) => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value));
  const fmtSeconds = (value: number | null | undefined) => (!isNum(value) ? '—' : value >= 60 ? `${(value / 60).toFixed(1)} 分` : `${value.toFixed(0)} 秒`);
  const fmtDateTime = (value: string | null | undefined) => {
    if (!value) return '—';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };
  // Loading placeholder that turns into the failure text once a load has failed.
  const pending = () => (s.error ? emptyState(`无法读取：${s.error}`, 'AlertTriangle') : emptyState('正在读取…', 'Loader2'));

  function processChip(process: ProcessStatus) {
    if (process.state === 'running') return '<span class="mini-pill active"><span data-icon="Loader2"></span>运行中</span>';
    if (process.state === 'stopped') return '<span class="mini-pill pending">已停止</span>';
    return '<span class="mini-pill idle">未启动</span>';
  }

  const CELL_STATE: Record<CellState, { label: string; tone: string }> = {
    pending: { label: '待执行', tone: 'idle' },
    running: { label: '运行中', tone: 'active' },
    done: { label: '已完成', tone: 'ready' },
    failed: { label: '失败', tone: 'failed' },
  };

  function cellChip(state: CellState) {
    const { label, tone } = CELL_STATE[state];
    return `<span class="mini-pill ${tone}">${esc(label)}</span>`;
  }

  function progressBar(done: number, total: number | null, extra = '') {
    const percent = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    return `<div class="exp-progress ${extra}"><div class="exp-progress-bar" style="width:${percent}%"></div></div>`;
  }

  // ------------------------------------------------------------ the cell matrix table (preview and live)

  // Tasks as rows, arms as columns; each cell holds one tile per seed.
  function matrixTable(tasks: string[], arms: string[], tiles: (task: string, arm: string) => string) {
    return `<div class="exp-table-scroll"><table class="exp-matrix"><thead><tr><th class="exp-matrix-corner">任务 \\ 反馈臂</th>${arms.map((arm) => `<th><span>${esc(arm)}</span><small>${esc(armLabel(arm))}</small></th>`).join('')}</tr></thead>
      <tbody>${tasks.map((task) => `<tr><th>${esc(task)}</th>${arms.map((arm) => `<td><div class="exp-tiles">${tiles(task, arm) || '<span class="meta">—</span>'}</div></td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  // ------------------------------------------------------------ page 1: start panel and run list

  const defaultRunName = (spec: SpecSummary) => spec.name.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[^A-Za-z0-9]+/, '') || 'run';

  function renderStart(target: HTMLElement) {
    const specs = s.specs;
    const heading = '<div class="panel-heading"><span data-icon="Play"></span><span>启动新运行</span></div>';
    if (!specs) {
      target.innerHTML = s.specsError
        ? `${heading}${emptyState(`无法读取规格：${s.specsError}`, 'AlertTriangle')}`
        : `${heading}${emptyState('正在读取规格…', 'Loader2')}`;
      return;
    }
    if (!specs.length) {
      target.innerHTML = `${heading}${emptyState('experiments/specs 下没有可用的实验方案。', 'Info')}`;
      return;
    }
    // Newest registered spec first, so the current experiment is the default.
    const ordered = [...specs].sort((a, b) =>
      String(b.registeredAt || '').localeCompare(String(a.registeredAt || '')) || a.name.localeCompare(b.name));
    if (!ordered.some((spec) => spec.file === s.form.spec)) pickSpec(ordered[0]);
    const spec = ordered.find((item) => item.file === s.form.spec) as SpecSummary;
    const perArm = spec.tasks.length * spec.seeds.length;
    const armValues = spec.arms.map((arm) => `arm=${arm}`);
    const custom = Boolean(s.form.only) && !armValues.includes(s.form.only);
    const scopeChip = (value: string, label: string, active: boolean) =>
      `<button type="button" class="exp-choice${active ? ' active' : ''}" data-exp-scope="${esc(value)}">${esc(label)}</button>`;
    target.innerHTML = `
      ${heading}
      <p class="exp-section-lead">方案决定要跑的任务、反馈臂、种子和每格轮数。启动后每格独立运行，互不影响。</p>
      <label class="exp-field"><span>实验方案</span>
        <select id="expSpec">${ordered.map((item) => `<option value="${esc(item.file)}"${item.file === spec.file ? ' selected' : ''}>${esc(item.name)} · ${esc(item.cellCount)} 格</option>`).join('')}</select>
      </label>
      <div class="exp-spec-summary">
        <div class="exp-summary-head"><strong>${esc(spec.name)}</strong><span class="exp-tag">${esc(spec.cellCount)} 个单元格</span></div>
        <dl class="exp-facts">
          <div><dt>任务</dt><dd>${spec.tasks.map(esc).join('、')}</dd></div>
          <div><dt>反馈臂</dt><dd class="exp-arm-list">${spec.arms.map((arm) => `<span class="exp-arm"><b>${esc(arm)}</b><small>${esc(armLabel(arm))}</small></span>`).join('')}</dd></div>
          <div><dt>种子</dt><dd>${esc(seedList(spec.seeds))}</dd></div>
          <div><dt>每格预算</dt><dd>${esc(spec.budget)} 轮</dd></div>
        </dl>
        ${spec.description ? `<details class="exp-desc"><summary>方案说明</summary><p>${esc(spec.description)}</p></details>` : ''}
      </div>
      <label class="exp-field"><span>运行名称</span><input id="expName" type="text" autocomplete="off" spellcheck="false" value="${esc(s.form.name)}" /></label>
      <div id="expNameHint" class="exp-hint"></div>
      <div class="exp-field"><span>运行范围</span>
        <div class="exp-choice-row">
          ${scopeChip('', `全部 ${spec.cellCount} 格`, !s.form.only)}
          ${spec.arms.map((arm) => scopeChip(`arm=${arm}`, `只跑 ${arm}（${perArm} 格）`, s.form.only === `arm=${arm}`)).join('')}
          ${scopeChip('__custom', '自定义…', custom)}
        </div>
      </div>
      <details class="exp-advanced exp-keep"${custom ? ' open' : ''}>
        <summary>高级：手动筛选单元格</summary>
        <label class="exp-field"><span>只跑（--only）</span><input id="expOnly" type="text" autocomplete="off" spellcheck="false" placeholder="如 task=real-weather,arm=scalar" value="${esc(s.form.only)}" /></label>
        <p class="exp-hint">多个条件用逗号分隔，条件之间取并集；字段可用 task、arm、seed、cell。留空即全部 ${esc(spec.cellCount)} 格。</p>
      </details>
      <div class="exp-field-row">
        <label class="exp-field"><span>并行进程数</span><input id="expWorkers" type="number" min="1" max="16" step="1" value="${esc(s.form.workers)}" /></label>
        <p class="exp-hint exp-workers-hint">每个单元格是一个训练进程。本机 10 核，建议 3–6，留出余量给 LLM 往返和系统。</p>
      </div>
      <div id="expStartError" class="exp-hint error"></div>
      <button id="expStartBtn" type="button" class="exp-start-btn" data-exp-action="start"><span data-icon="Play"></span><span>启动运行</span></button>
    `;
  }

  function pickSpec(spec: SpecSummary) {
    s.form.spec = spec.file;
    if (!s.form.nameTouched) s.form.name = defaultRunName(spec);
  }

  function nameProblem(): { text: string; blocking: boolean } {
    const name = s.form.name.trim();
    if (!name) return { text: '请填写运行名称。', blocking: true };
    if (!NAME_RE.test(name)) return { text: '名称只允许字母、数字、点、下划线和连字符，且必须以字母或数字开头。', blocking: true };
    if (!s.runs) return { text: '运行列表尚未读取，无法检查同名运行。', blocking: false };
    const existing = s.runs.find((run) => run.name === name);
    if (existing?.process.state === 'running') return { text: '该运行正在运行，请换一个名称或先停止它。', blocking: true };
    if (existing) return { text: `同名运行已存在（${existing.cellsDone}/${existing.cellsTotal ?? '?'} 格已完成）：启动会续跑并跳过已完成的单元格；一个运行目录只对应一个实验方案，换方案会被拒绝。`, blocking: false };
    return { text: '将创建新的运行目录。', blocking: false };
  }

  const validWorkers = (value: number) => Number.isInteger(value) && value >= 1 && value <= 16;

  function updateStartHint() {
    const hint = el('expNameHint');
    const button = el('expStartBtn') as HTMLButtonElement | null;
    if (!hint || !button) return;
    const problem = nameProblem();
    hint.textContent = problem.text;
    hint.classList.toggle('error', problem.blocking);
    const startError = el('expStartError');
    if (startError) startError.textContent = s.startError;
    button.disabled = problem.blocking || s.starting || !validWorkers(s.form.workers);
  }

  // Keep the scope chips in sync when the operator types a custom filter.
  function syncScopeChips() {
    const spec = s.specs?.find((item) => item.file === s.form.spec);
    const armValues = (spec?.arms ?? []).map((arm) => `arm=${arm}`);
    const custom = Boolean(s.form.only) && !armValues.includes(s.form.only);
    root.querySelectorAll<HTMLElement>('[data-exp-scope]').forEach((button) => {
      const value = button.dataset.expScope ?? '';
      button.classList.toggle('active', value === '__custom' ? custom : value === s.form.only);
    });
  }

  async function startRun() {
    const spec = s.specs?.find((item) => item.file === s.form.spec);
    if (!spec) return;
    const name = s.form.name.trim();
    s.starting = true;
    s.startError = '';
    updateStartHint();
    try {
      await postJson(`${API}/runs/${encodeURIComponent(name)}/start`, {
        spec: spec.file,
        workers: s.form.workers,
        only: s.form.only.trim(),
      });
      s.starting = false;
      navigate(pathFor({ run: name, cell: null, round: null }));
    } catch (caught) {
      s.starting = false;
      s.startError = errText(caught);
      updateStartHint();
    }
  }

  async function stopRun() {
    const name = s.route.run;
    if (!name) return;
    if (!confirm('停止会终止该运行的 batch 进程及其单元格进程；已完成的结果保留，之后可用同一个运行名称重新启动续跑。是否继续？')) return;
    await deps.runAction(async () => {
      await postJson(`${API}/runs/${encodeURIComponent(name)}/stop`, {});
    });
    await refreshNow();
  }

  // Restart the run's own spec: all cells ("续跑") or one cell ("单独运行此单元格").
  async function restartRun(only: string, workers: number) {
    const name = s.route.run;
    const specFile = s.run?.process.specFile;
    if (!name || !specFile) return;
    await deps.runAction(async () => {
      await postJson(`${API}/runs/${encodeURIComponent(name)}/start`, { spec: specFile, workers, only });
    });
    await refreshNow();
  }

  async function generateAnalysis() {
    const name = s.route.run;
    if (!name) return;
    s.analysisBusy = true;
    renderPage();
    await deps.runAction(async () => {
      await postJson(`${API}/runs/${encodeURIComponent(name)}/analysis`, {});
    });
    s.analysisBusy = false;
    s.analysisForce = true;
    await refreshNow();
  }

  function runRow(run: RunSummary) {
    const spec = run.process.specFile ? run.process.specFile.replace(/\.json$/, '') : '';
    return `<button type="button" class="exp-run-row" data-exp-go="${esc(pathFor({ run: run.name, cell: null, round: null }))}">
      <span class="exp-run-main">
        <strong>${esc(run.name)}</strong>
        <span class="meta">${spec ? `${esc(spec)} · ` : ''}最近活动 ${esc(fmtDateTime(run.lastActivityAt))}</span>
      </span>
      <span class="exp-run-progress">
        ${progressBar(run.cellsDone, run.cellsTotal)}
        <span class="meta">${esc(run.cellsDone)}/${esc(run.cellsTotal ?? '?')} 格</span>
      </span>
      ${processChip(run.process)}
      <span data-icon="ChevronRight"></span>
    </button>`;
  }

  function renderRuns(target: HTMLElement) {
    const heading = '<div class="panel-heading"><span data-icon="GitBranch"></span><span>运行记录</span></div>';
    const runs = s.runs;
    if (!runs) {
      target.innerHTML = s.runsError
        ? `${heading}${emptyState(`无法读取运行列表：${s.runsError}`, 'AlertTriangle')}`
        : `${heading}${emptyState('正在读取运行列表…', 'Loader2')}`;
      return;
    }
    if (!runs.length) {
      target.innerHTML = `${heading}${emptyState('还没有运行。在左侧选好方案后点“启动运行”。', 'Info')}`;
      return;
    }
    const byRecent = (a: RunSummary, b: RunSummary) =>
      String(b.lastActivityAt || '').localeCompare(String(a.lastActivityAt || '')) || a.name.localeCompare(b.name);
    const running = runs.filter((run) => run.process.state === 'running').sort(byRecent);
    const recent = runs.filter((run) => run.process.state !== 'running' && (run.cellsDone > 0 || run.lastActivityAt)).sort(byRecent);
    const empties = runs.filter((run) => run.process.state !== 'running' && run.cellsDone === 0 && !run.lastActivityAt).sort((a, b) => a.name.localeCompare(b.name));
    const group = (title: string, items: RunSummary[]) =>
      items.length ? `<div class="exp-group-title">${esc(title)}<span class="exp-group-count">${items.length}</span></div><div class="exp-run-list">${items.map(runRow).join('')}</div>` : '';
    target.innerHTML = `${heading}
      ${running.length + recent.length ? '' : emptyState('还没有运行。在左侧选好方案后点“启动运行”。', 'Info')}
      ${group('正在运行', running)}
      ${group('最近运行', recent)}
      ${empties.length ? `<details class="exp-empty-runs exp-keep"><summary>没有运行记录的目录（${empties.length}）</summary>
        <p class="exp-hint">这些目录里没有单元格或批次记录（例如 stage0 的输出目录）。点开仍可查看。</p>
        <div class="exp-empty-list">${empties.map((run) => `<button type="button" class="exp-empty-row" data-exp-go="${esc(pathFor({ run: run.name, cell: null, round: null }))}">${esc(run.name)}<span data-icon="ChevronRight"></span></button>`).join('')}</div>
      </details>` : ''}
      ${s.runsError ? `<p class="exp-hint error">刷新运行列表失败：${esc(s.runsError)}</p>` : ''}`;
  }

  // ------------------------------------------------------------ page 2: run view

  function runTotals(run: RunSnapshot) {
    const count = (state: CellState) => run.cells.filter((cell) => cell.state === state).length;
    return {
      done: count('done'),
      running: count('running'),
      failed: count('failed'),
      total: run.spec?.cellCount ?? run.cells.length,
    };
  }

  function renderRunHead(target: HTMLElement) {
    const run = s.run;
    if (!run) {
      target.innerHTML = pending();
      return;
    }
    const totals = runTotals(run);
    const process = run.process;
    const canResume = process.state !== 'running' && Boolean(process.specFile) && totals.done < totals.total;
    const stoppedPartial = process.state === 'stopped' && totals.done < totals.total;
    const finished = totals.total > 0 && totals.done >= totals.total;
    const spec = run.spec;
    target.innerHTML = `
      <div class="exp-head-row">
        <div class="exp-head-title">
          <h3>${esc(run.name)}</h3>
          ${processChip(process)}
        </div>
        <div class="exp-nav-btns">
          ${process.state === 'running' ? '<button type="button" class="danger ghost" data-exp-action="stop"><span data-icon="Square"></span><span>停止</span></button>' : ''}
          ${canResume ? '<button type="button" class="secondary" data-exp-action="resume"><span data-icon="Play"></span><span>继续运行</span></button>' : ''}
        </div>
      </div>
      <div class="meta exp-head-meta">
        ${spec ? `方案 ${esc(spec.name)} · ` : ''}${process.specFile ? `${esc(process.specFile)} · ` : ''}${process.pid ? `pid ${esc(process.pid)} · ` : ''}启动于 ${esc(fmtDateTime(process.startedAt))}
      </div>
      ${spec ? `<div class="exp-spec-line">${esc(spec.tasks.length)} 任务 · ${esc(spec.arms.length)} 个反馈臂 · ${esc(spec.seeds.length)} 个种子 · 每格 ${esc(spec.budget)} 轮</div>` : ''}
      ${progressBar(totals.done, totals.total, 'large')}
      <div class="exp-counts">
        <span class="count-done"><strong>${esc(totals.done)}</strong><small>已完成 / ${esc(totals.total)}</small></span>
        <span class="count-running"><strong>${esc(totals.running)}</strong><small>运行中</small></span>
        <span class="count-failed"><strong>${esc(totals.failed)}</strong><small>失败</small></span>
        <span><strong>${esc(Math.max(0, totals.total - totals.done - totals.running - totals.failed))}</strong><small>待执行</small></span>
      </div>
      ${stoppedPartial ? '<div class="exp-notice"><span data-icon="Info"></span><p>运行已停止。已完成的结果都保留着；点“继续运行”会跳过已完成的单元格，从未完成的格子接着跑。</p></div>' : ''}
      ${finished && process.state !== 'running' ? '<div class="exp-notice ok"><span data-icon="Check"></span><p>所有单元格都已完成。可以在下方生成分析。</p></div>' : ''}`;
  }

  function orderedAxes(run: RunSnapshot) {
    const unique = (values: string[]) => [...new Set(values)];
    return {
      tasks: run.spec?.tasks.length ? run.spec.tasks : unique(run.cells.map((cell) => cell.task)),
      arms: run.spec?.arms.length ? run.spec.arms : unique(run.cells.map((cell) => cell.arm)),
    };
  }

  function renderMatrix(target: HTMLElement) {
    const run = s.run;
    const heading = '<div class="panel-heading"><span data-icon="Boxes"></span><span>单元格矩阵</span></div><p class="exp-section-lead">行是任务，列是反馈臂，每一格方块是一个种子。点方块进入单元格。</p>';
    if (!run) {
      target.innerHTML = heading;
      return;
    }
    if (!run.cells.length) {
      target.innerHTML = `${heading}${emptyState('这个运行还没有单元格记录。', 'Info')}`;
      return;
    }
    const { tasks, arms } = orderedAxes(run);
    const tile = (cell: CellSummary) => {
      const pct = cell.budget ? Math.min(100, Math.round((cell.roundsDone / cell.budget) * 100)) : 0;
      const sat = cell.saturated === null ? '' : `<span class="exp-sat ${cell.saturated ? 'yes' : 'no'}" title="${cell.saturated ? '已饱和：认证认为没有更多可改进空间' : '未饱和'}"></span>`;
      const tip = `${cell.cell}\n${CELL_STATE[cell.state].label} · ${cell.roundsDone}/${cell.budget ?? '?'} 轮 · 接受 ${cell.accepted}\n当前最优候选 ${cell.incumbent}\n诊断 MSE ${fmtMse(cell.incumbentMse)}`;
      const incumbent = cell.incumbent ? `<span class="exp-tile-inc" title="当前最优候选 ${esc(cell.incumbent)}">${esc(cell.incumbent)}</span>` : '';
      return `<button type="button" class="exp-tile ${cell.state}" style="--p:${pct}%" title="${esc(tip)}" data-exp-go="${esc(pathFor({ run: run.name, cell: cell.cell, round: null }))}">
        ${sat}<span class="exp-tile-seed">s${esc(cell.seed)}</span><span class="exp-tile-rounds">${esc(cell.roundsDone)}/${esc(cell.budget ?? '?')} · ✓${esc(cell.accepted)}</span>${incumbent}</button>`;
    };
    target.innerHTML = `${heading}
      ${matrixTable(tasks, arms, (task, arm) => run.cells.filter((cell) => cell.task === task && cell.arm === arm).sort((a, b) => a.seed - b.seed).map(tile).join(''))}
      <div class="exp-legend">
        <span><i class="exp-swatch pending"></i>待执行</span><span><i class="exp-swatch running"></i>运行中</span>
        <span><i class="exp-swatch done"></i>已完成</span><span><i class="exp-swatch failed"></i>失败</span>
        <span><i class="exp-sat yes inline"></i>已饱和</span><span><i class="exp-sat no inline"></i>未饱和</span>
        <span class="meta">方块内：种子 · 已跑轮数/总轮数 · ✓ 接受的候选数 · 当前最优候选；悬停看诊断 MSE。</span>
      </div>`;
  }

  // posthoc analysis (experiments/posthoc.py analysis.json), rendered as the server returns it.
  function renderAnalysis(target: HTMLElement) {
    const data = s.analysis;
    const analysis: Analysis | null = data?.analysis ?? null;
    const running = s.analysisBusy || data?.process.state === 'running';
    const failed = !running && !analysis && data?.process.state === 'stopped';
    const chip = running
      ? '<span class="mini-pill active"><span data-icon="Loader2"></span>生成中</span>'
      : failed ? '<span class="mini-pill failed">失败</span>'
      : analysis ? '<span class="mini-pill ready">已生成</span>' : '<span class="mini-pill idle">未生成</span>';
    const names = (list: string[] | undefined) => (list ?? []).map(esc).join('、');
    const parts: string[] = [];
    if (s.analysisError) parts.push(`<div class="exp-hint error">读取分析失败：${esc(s.analysisError)}</div>`);
    if (!analysis) {
      parts.push(emptyState(
        running ? '正在生成分析…'
          : failed ? '分析进程已结束但没有写出 analysis.json，原因见下方日志。'
          : '尚未生成。点击“生成分析”，对本运行中已完成的单元格做 posthoc 分析。',
        running ? 'Loader2' : failed ? 'AlertTriangle' : 'Info',
      ));
    } else {
      const records = analysis.runs ?? [];
      const broken = records.filter((record) => record.error);
      const kinds = [...new Set(records.map((record) => record.e1Kind).filter(Boolean))];
      const stat = (value: { median?: number | null; iqr?: number | null; q25?: number | null; q75?: number | null; n?: number } | undefined) => ({
        median: fmtMse(value?.median),
        iqr: fmtMse(value?.iqr),
        range: `Q25 ${fmtMse(value?.q25)} · Q75 ${fmtMse(value?.q75)}`,
        n: value?.n ?? '—',
      });
      const rows = analysis.endpointTable ?? [];
      const contrasts = analysis.contrasts ?? [];
      parts.push(`<div class="meta exp-head-meta">已分析 ${esc(records.length - broken.length)} 个单元格${running ? '（正在重新生成，下表是上一次的结果）' : ''}${kinds.length ? ` · E1 类型 ${names(kinds as string[])}` : ''}</div>`);
      if (analysis.incomplete?.length) parts.push(`<div class="exp-hint">未完成、未纳入分析的单元格（${esc(analysis.incomplete.length)}）：${names(analysis.incomplete)}</div>`);
      if (broken.length) parts.push(`<div class="exp-hint error">分析出错的单元格：${broken.map((record) => `${esc(record.run ?? '?')}（${esc(record.error)}）`).join('、')}</div>`);
      parts.push('<div class="detail-section-title">端点表（按任务 × 反馈臂）</div>');
      parts.push('<p class="exp-hint">E1 = 最终最优候选相对 m0 基线的归一化误差（真实任务是 MSE 比，合成任务是最优性间隙比），小于 1 表示优于基线；E2 = 各轮该比值的均值（曲线下面积）。两个指标都是越小越好。</p>');
      parts.push(rows.length
        ? `<div class="exp-table-scroll"><table class="exp-table exp-endpoints">
            <thead><tr><th>任务</th><th>反馈臂</th><th>单元格数</th><th>E1 中位</th><th>E1 IQR</th><th>E2 中位</th><th>E2 IQR</th><th>n（E1/E2）</th></tr></thead>
            <tbody>${rows.map((row) => {
              const e1 = stat(row.e1);
              const e2 = stat(row.e2);
              return `<tr><td>${esc(row.task ?? '—')}</td><td>${esc(row.arm ?? '—')}</td><td class="num">${esc(row.nRuns ?? '—')}</td>
                <td class="num">${esc(e1.median)}</td><td class="num" title="${esc(e1.range)}">${esc(e1.iqr)}</td>
                <td class="num">${esc(e2.median)}</td><td class="num" title="${esc(e2.range)}">${esc(e2.iqr)}</td>
                <td class="num">${esc(e1.n)}/${esc(e2.n)}</td></tr>`;
            }).join('')}</tbody></table></div>`
        : emptyState('没有可汇总的已完成运行。', 'Info'));
      parts.push('<div class="detail-section-title">反馈臂对比</div>');
      parts.push('<p class="exp-hint">以（任务, 种子）为区块，对 E1 的均值差做符号翻转置换检验；统计量为正表示前者误差更大。</p>');
      parts.push(contrasts.length
        ? `<div class="exp-table-scroll"><table class="exp-table exp-contrasts">
            <thead><tr><th>对比</th><th>区块数</th><th>统计量（E1 均值差）</th><th>p 值</th><th>置换分布 2.5% – 97.5%</th><th>置换方式</th></tr></thead>
            <tbody>${contrasts.map((row) => `<tr><td>${esc(row.contrast ?? '—')}</td><td class="num">${esc(row.nBlocks ?? '—')}</td>
              <td class="num">${esc(fmtSigned(row.statistic))}</td><td class="num">${esc(fmtP(row.pValue))}</td>
              <td class="num">${esc(fmtSigned(row.permLow))} … ${esc(fmtSigned(row.permHigh))}</td><td>${esc(row.permutations ?? '—')}</td></tr>`).join('')}</tbody></table></div>`
        : emptyState('没有对比结果。', 'Info'));
      const audit = analysis.audit;
      if (audit?.pathViolationRuns?.length) parts.push(`<div class="exp-hint error">审计：路径违规的运行 ${names(audit.pathViolationRuns)}</div>`);
      if (audit?.gainRatioFlagRuns?.length) parts.push(`<div class="exp-hint error">审计：增益比标记的运行 ${names(audit.gainRatioFlagRuns)}</div>`);
    }
    if (data?.logTail) {
      parts.push(`<details class="exp-keep exp-sublog"${failed ? ' open' : ''}><summary>分析日志尾部</summary><pre class="exp-pre">${esc(data.logTail)}</pre></details>`);
    }
    target.innerHTML = `
      <div class="exp-head-row exp-analysis-head">
        <div class="exp-head-title panel-heading"><span data-icon="Gauge"></span><span>留出集分析（posthoc）</span>${chip}</div>
        <button type="button" class="secondary" data-exp-action="analyze"${running ? ' disabled' : ''}><span data-icon="Gauge"></span><span>${analysis ? '重新生成分析' : '生成分析'}</span></button>
      </div>
      <p class="exp-section-lead">对已完成的单元格重放最优候选，在留出的测试段上计算论文用的端点指标，不调用 LLM。</p>
      ${parts.join('')}`;
  }

  function eventRow(event: BatchEvent) {
    return `<tr><td>${esc(fmtDateTime(event.at))}</td><td>${esc(event.cell)}</td>
      <td><span class="mini-pill ${event.event === 'ok' ? 'ready' : 'failed'}">${esc(event.event)}</span></td>
      <td class="num">${esc(event.exitCode ?? '—')}</td><td class="num">${esc(event.attempt)}</td><td class="num">${esc(fmtSeconds(event.seconds))}</td>
      <td class="exp-ellipsis" title="${esc(event.detail || '')}">${esc(event.detail || '')}</td></tr>`;
  }

  function renderRunLog(target: HTMLElement) {
    const run = s.run;
    if (!run) return;
    const events = run.batchEvents.slice(-30).reverse();
    target.innerHTML = `
      <div class="detail-section-title">最近事件</div>
      ${events.length ? `<div class="exp-table-scroll"><table class="exp-table"><thead><tr><th>时间</th><th>单元格</th><th>事件</th><th>退出码</th><th>尝试</th><th>耗时</th><th>详情</th></tr></thead><tbody>${events.map(eventRow).join('')}</tbody></table></div>` : emptyState('还没有 batch 事件。', 'Info')}
      <div class="detail-section-title">batch 日志尾部</div>
      <pre class="exp-pre">${esc(run.batchLogTail || '（空）')}</pre>`;
  }

  // ------------------------------------------------------------ page 3: cell view

  // The incumbent's diag MSE after the last round (the curve carries it forward on rejections).
  function currentIncumbentMse(cell: CellDetail): number | null {
    for (let i = cell.incumbentCurve.length - 1; i >= 0; i -= 1) {
      if (isNum(cell.incumbentCurve[i].mse)) return cell.incumbentCurve[i].mse;
    }
    return null;
  }

  function renderCellHead(target: HTMLElement) {
    const cell = s.cell;
    if (!cell) {
      target.innerHTML = pending();
      return;
    }
    const acceptedRows = cell.rounds.filter((row) => row.accepted);
    const accepted = acceptedRows.length;
    const incumbent = acceptedRows.length ? acceptedRows[acceptedRows.length - 1].candidateId : 'm0（基线）';
    const process = s.run && s.run.name === s.route.run ? s.run.process : null;
    const canRun = Boolean(process && process.state !== 'running' && process.specFile) && cell.state !== 'done';
    target.innerHTML = `
      <div class="exp-head-row">
        <div class="exp-head-title"><h3>${esc(cell.cell)}</h3>${cellChip(cell.state)}</div>
        <div class="exp-nav-btns">
          ${canRun ? '<button type="button" class="secondary" data-exp-action="runcell"><span data-icon="Play"></span><span>单独运行此单元格</span></button>' : ''}
          <button type="button" class="secondary" data-exp-go="${esc(pathFor({ run: s.route.run, cell: null, round: null }))}"><span data-icon="ChevronLeft"></span><span>返回运行</span></button>
        </div>
      </div>
      <div class="meta exp-head-meta">任务 ${esc(cell.task)} · 反馈臂 ${esc(cell.arm)}（${esc(armLabel(cell.arm))}） · 种子 ${esc(cell.seed)}</div>
      ${progressBar(cell.rounds.length, cell.budget, 'large')}
      <div class="exp-counts">
        <span><strong>${esc(cell.rounds.length)}/${esc(cell.budget ?? '?')}</strong><small>已跑轮数</small></span>
        <span class="count-done"><strong>${esc(accepted)}</strong><small>接受的候选</small></span>
        <span><strong>${esc(fmtMse(currentIncumbentMse(cell)))}</strong><small>当前最优诊断 MSE</small></span>
        <span class="count-wide"><strong>${esc(incumbent)}</strong><small>当前最优候选</small></span>
      </div>`;
  }

  function renderChart(target: HTMLElement) {
    const cell = s.cell;
    const heading = '<div class="panel-heading"><span data-icon="Activity"></span><span>诊断 MSE 随轮次变化</span></div><p class="exp-section-lead">曲线是当前最优候选在诊断段上的分数；被接受时下降，被拒绝时持平。</p>';
    if (!cell) {
      target.innerHTML = heading;
      return;
    }
    // Only the incumbent curve is drawn: round 0 is the m0 baseline, the line steps down at
    // accepted rounds and stays flat otherwise.
    const line = cell.incumbentCurve.filter((point): point is { round: number; mse: number; accepted: boolean } => isNum(point.mse));
    if (!line.length) {
      target.innerHTML = `${heading}${emptyState('还没有可绘制的曲线。', 'Info')}`;
      return;
    }
    const lo = Math.min(...line.map((p) => p.mse));
    const hi = Math.max(...line.map((p) => p.mse));
    const span = hi - lo || Math.abs(hi) * 0.1 || 1;
    const yMin = lo - span * 0.12;
    const yMax = hi + span * 0.12;
    const W = 760, H = 280, L = 62, R = 18, T = 16, B = 38;
    const lastRound = line[line.length - 1].round;
    const xMax = Math.max(cell.budget ?? 0, lastRound, 1);
    const x = (round: number) => L + (round / xMax) * (W - L - R);
    const y = (value: number) => T + (1 - (value - yMin) / (yMax - yMin)) * (H - T - B);
    const go = (round: number) => esc(pathFor({ run: s.route.run, cell: cell.cell, round }));
    const yTicks = [0, 1, 2, 3].map((i) => yMin + ((yMax - yMin) * i) / 3);
    const xStep = xMax <= 12 ? 1 : xMax <= 30 ? 5 : 10;
    const xTicks: number[] = [];
    for (let r = 0; r <= xMax; r += 1) if (r === 0 || r % xStep === 0) xTicks.push(r);
    const path = line.map((point, index) => (index === 0 ? `M${x(point.round)},${y(point.mse)}` : `H${x(point.round)}V${y(point.mse)}`)).join('') + `H${x(lastRound)}`;
    const pointTip = (point: { round: number; mse: number }) => {
      const row = cell.rounds.find((item) => item.round === point.round);
      return `第 ${point.round} 轮${row ? ` · ${row.candidateId}` : ''}\n最优候选诊断 MSE ${fmtMse(point.mse)} · 已接受`;
    };
    target.innerHTML = `${heading}
      <svg class="exp-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="诊断 MSE 随轮次变化">
        ${yTicks.map((tick) => `<line class="metric-grid-line" x1="${L}" x2="${W - R}" y1="${y(tick)}" y2="${y(tick)}"></line><text class="exp-axis-text y" x="${L - 8}" y="${y(tick) + 4}">${esc(fmtMse(tick))}</text>`).join('')}
        <line class="metric-axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"></line>
        ${xTicks.map((tick) => `<line class="metric-tick" x1="${x(tick)}" x2="${x(tick)}" y1="${H - B}" y2="${H - B + 5}"></line><text class="exp-axis-text x" x="${x(tick)}" y="${H - B + 19}">${tick}</text>`).join('')}
        <text class="exp-axis-text x" x="${(L + W - R) / 2}" y="${H - 4}">轮次</text>
        ${cell.certifications.filter((c) => c.saturated).map((c) => `<line class="exp-sat-line" x1="${x(c.round)}" x2="${x(c.round)}" y1="${T}" y2="${H - B}"><title>第 ${esc(c.round)} 轮认证：已饱和</title></line>`).join('')}
        <path class="exp-best-line" d="${path}"></path>
        ${line.filter((point) => point.round === 0).map((point) => `<circle class="exp-pt base" cx="${x(0)}" cy="${y(point.mse)}" r="4.5"><title>m0 基线 · diag MSE ${esc(fmtMse(point.mse))}</title></circle>`).join('')}
        ${line.filter((point) => point.round > 0 && point.accepted).map((point) => `<circle class="exp-pt accept" data-exp-go="${go(point.round)}" cx="${x(point.round)}" cy="${y(point.mse)}" r="5.5"><title>${esc(pointTip(point))}</title></circle>`).join('')}
      </svg>
      <div class="exp-legend">
        <span><i class="exp-line-swatch"></i>最优候选的诊断 MSE</span><span><i class="exp-dot base"></i>m0 基线</span>
        <span><i class="exp-dot accept"></i>接受新候选</span>
        <span><i class="exp-line-swatch sat"></i>认证判饱和的轮次</span><span class="meta">点击绿点进入该轮</span>
      </div>`;
  }

  function statusChip(row: RoundRow) {
    const label = row.status === 'evaluated' ? '已评估' : row.status === 'failed' ? '失败' : row.status;
    const tone = row.status === 'evaluated' ? 'ready' : /fail|error|invalid/.test(row.status) ? 'failed' : 'pending';
    return `<span class="mini-pill ${tone}">${esc(label)}</span>`;
  }

  function renderRounds(target: HTMLElement) {
    const cell = s.cell;
    const heading = '<div class="panel-heading"><span data-icon="GitBranch"></span><span>逐轮结果</span></div><p class="exp-section-lead">点击任意一行进入该轮，查看提案和智能体会话。诊断 MSE 是智能体当轮看到的分数；论文数字以留出集分析为准。</p>';
    if (!cell || !cell.rounds.length) {
      target.innerHTML = `${heading}${emptyState('还没有轮次记录。', 'Info')}`;
      return;
    }
    const body = cell.rounds.map((row) => {
      const total = row.tokens.inputTokens + row.tokens.outputTokens;
      const tip = `输入 ${row.tokens.inputTokens} · 输出 ${row.tokens.outputTokens} · 缓存读 ${row.tokens.cacheReadTokens} · 推理 ${row.tokens.reasoningTokens}`;
      const judge = row.judgeCause
        ? `${esc(row.judgeCause)}${row.judgeFailed ? ' <span class="mini-pill failed">失败</span>' : ''}${row.inspirationParent ? `<small class="meta"> ← ${esc(row.inspirationParent)}</small>` : ''}`
        : '—';
      return `<tr class="${row.accepted ? 'accepted' : ''}" tabindex="0" data-exp-go="${esc(pathFor({ run: s.route.run, cell: cell.cell, round: row.round }))}">
        <td class="num">${esc(row.round)}</td>
        <td><div class="exp-cand">${esc(row.candidateId)}</div>${row.targetAspect ? `<small class="meta">${esc(row.targetAspect)}</small>` : ''}</td>
        <td>${esc(row.parent)}</td><td>${statusChip(row)}</td>
        <td class="num">${esc(fmtMse(row.diagMse))}</td><td class="num">${esc(fmtP(row.dmP))}</td>
        <td class="center">${row.accepted ? '<span class="exp-ok">✓</span>' : ''}</td>
        <td>${judge}</td><td class="num" title="${esc(tip)}">${esc(fmtK(total))}</td><td class="num">${esc(fmtSeconds(row.seconds))}</td>
        <td class="exp-ellipsis" title="${esc(row.error || row.evalNote || '')}">${esc(row.error || '')}</td></tr>`;
    }).join('');
    target.innerHTML = `${heading}
      <div class="exp-table-scroll"><table class="exp-table exp-rounds">
        <thead><tr><th>轮次</th><th>候选模型</th><th>父候选</th><th>状态</th><th>诊断 MSE</th><th>显著性 p</th><th>接受</th><th>多样性判官</th><th>tokens</th><th>耗时</th><th>错误</th></tr></thead>
        <tbody>${body}</tbody></table></div>`;
  }

  function certRow(row: CertRow) {
    return `<div class="exp-cert ${row.saturated ? 'saturated' : ''}">
      <div class="exp-cert-head">
        <b>第 ${esc(row.round)} 轮</b><span class="meta">最优候选 ${esc(row.incumbentId)} · 水平 ${esc(fmtMse(row.level))}</span>
        <span class="mini-pill ${row.saturated ? 'ready' : 'idle'}">${row.saturated ? '已饱和' : '未饱和'}</span>
      </div>
      ${row.failures.length ? `<div class="exp-cert-fails"><small class="meta">未通过的统计量</small>${row.failures.map((f) => `<span class="exp-tag warn" title="UCB ${esc(fmtMse(f.ucb))} ≥ 阈值 ${esc(fmtMse(f.threshold))}">${esc(f.label)} <em>${esc(fmtMse(f.ucb))} / ${esc(fmtMse(f.threshold))}</em></span>`).join('')}</div>` : ''}
    </div>`;
  }

  function renderCerts(target: HTMLElement) {
    const cell = s.cell;
    const heading = '<div class="panel-heading"><span data-icon="ShieldCheck"></span><span>饱和认证时间线</span></div><p class="exp-section-lead">每次更换最优候选后做一次认证：所有统计量的增益上界都低于阈值时，判为饱和（没有更多可改进空间）。</p>';
    if (!cell || !cell.certifications.length) {
      target.innerHTML = `${heading}${emptyState('还没有认证记录。', 'Info')}`;
      return;
    }
    target.innerHTML = `${heading}<div class="exp-cert-list">${cell.certifications.map(certRow).join('')}</div>`;
  }

  function renderCellLog(target: HTMLElement) {
    target.innerHTML = `<pre class="exp-pre">${esc(s.cell?.logTail || '（空）')}</pre>`;
  }

  // ------------------------------------------------------------ page 4: round detail

  function roundRow(): RoundRow | null {
    return s.cell?.rounds.find((row) => row.round === s.route.round) || null;
  }

  function renderRoundHead(target: HTMLElement) {
    const n = s.route.round as number;
    const row = roundRow();
    const rounds = s.cell?.rounds.map((r) => r.round) || [];
    const prev = rounds.filter((r) => r < n).pop();
    const next = rounds.find((r) => r > n);
    const go = (round: number | undefined) => (round === undefined ? '' : `data-exp-go="${esc(pathFor({ run: s.route.run, cell: s.route.cell, round }))}"`);
    target.innerHTML = `
      <div class="exp-head-row">
        <div class="exp-head-title"><h3>第 ${esc(n)} 轮</h3>${row ? statusChip(row) : ''}${row?.accepted ? '<span class="mini-pill ready">已接受 ✓</span>' : ''}</div>
        <div class="exp-nav-btns">
          <button type="button" class="secondary" ${go(prev)} ${prev === undefined ? 'disabled' : ''}><span data-icon="ChevronLeft"></span><span>上一轮</span></button>
          <button type="button" class="secondary" ${go(next)} ${next === undefined ? 'disabled' : ''}><span>下一轮</span><span data-icon="ChevronRight"></span></button>
        </div>
      </div>
      ${row ? `<div class="meta exp-head-meta">候选模型 ${esc(row.candidateId)} · 父候选 ${esc(row.parent)} · 诊断 MSE ${esc(fmtMse(row.diagMse))} · 显著性 p ${esc(fmtP(row.dmP))} · 耗时 ${esc(fmtSeconds(row.seconds))}${row.error ? ` · 错误：${esc(row.error)}` : ''}</div>` : ''}`;
  }

  // Every key of the proposal object is shown; the four known fields come first, with their labels.
  const PROPOSAL_LABELS: Record<string, string> = {
    targetAspect: '方向（targetAspect）',
    aspectDetail: '细节（aspectDetail）',
    mechanism: '机制（mechanism）',
    expectation: '预期（expectation）',
  };

  function renderProposal(target: HTMLElement) {
    const heading = '<div class="panel-heading"><span data-icon="Sparkles"></span><span>提案</span></div>';
    const proposal = s.round?.proposal;
    if (!proposal) {
      target.innerHTML = `${heading}${pending()}`;
      return;
    }
    const known = Object.keys(PROPOSAL_LABELS).filter((key) => key in proposal);
    const keys = [...known, ...Object.keys(proposal).filter((key) => !known.includes(key))];
    const show = (value: any) => (typeof value === 'string' ? value || '—' : JSON.stringify(value, null, 2));
    target.innerHTML = `${heading}${keys.length
      ? `<dl class="exp-dl">${keys.map((key) => `<dt>${esc(PROPOSAL_LABELS[key] ?? key)}</dt><dd>${esc(show(proposal[key]))}</dd>`).join('')}</dl>`
      : emptyState('这一轮没有提案内容。', 'Info')}`;
  }

  function renderPosthoc(target: HTMLElement) {
    const heading = '<div class="panel-heading"><span data-icon="ShieldCheck"></span><span>留出集增益卡片</span></div><p class="exp-section-lead">在留出段上估计每个通道组的真实增益区间（Δ 与 95% 置信上下界），仅操作者可见。</p>';
    if (!s.round) {
      target.innerHTML = `${heading}${pending()}`;
      return;
    }
    const cards = s.round.posthoc;
    if (!cards || !cards.length) {
      target.innerHTML = `${heading}${emptyState('这一轮没有 post-hoc 卡片。', 'Info')}`;
      return;
    }
    target.innerHTML = `${heading}<div class="exp-table-scroll"><table class="exp-table">
      <thead><tr><th>统计量</th><th>方面</th><th>Δ</th><th>LCB</th><th>UCB</th></tr></thead>
      <tbody>${cards.map((card) => `<tr><td>${esc(card.label)}</td><td>${esc(card.aspect)}</td><td class="num">${esc(fmtSigned(card.delta))}</td><td class="num">${esc(fmtSigned(card.lcb))}</td><td class="num">${esc(fmtSigned(card.ucb))}</td></tr>`).join('')}</tbody></table></div>`;
  }

  function renderSession(target: HTMLElement) {
    const heading = '<div class="panel-heading"><span data-icon="Bot"></span><span>智能体会话（公开部件）</span></div>';
    if (!s.round) {
      target.innerHTML = `${heading}${pending()}`;
      return;
    }
    const parts = deps.normalizeChatParts(s.round.parts.map((part) => ({ ...part, sourceLabel: '智能体' })));
    if (!parts.length) {
      target.innerHTML = `${heading}${emptyState('这一轮没有可显示的会话记录。', 'Info')}`;
      return;
    }
    target.innerHTML = `${heading}<div class="builder-trace exp-session">${parts.map((part) => deps.messageHtml(part)).join('')}</div>`;
    deps.bindToolCards(target);
  }

  // ------------------------------------------------------------ events

  root.addEventListener('click', (event) => {
    const target = event.target as Element;
    const action = target.closest<HTMLElement>('[data-exp-action]');
    if (action) {
      if ((action as HTMLButtonElement).disabled) return;
      const name = action.dataset.expAction;
      if (name === 'settings') deps.goSettings();
      else if (name === 'start') void startRun();
      else if (name === 'stop') void stopRun();
      else if (name === 'resume') void restartRun('', validWorkers(s.form.workers) ? s.form.workers : 4);
      else if (name === 'runcell') void restartRun(`cell=${s.route.cell}`, 1);
      else if (name === 'analyze') void generateAnalysis();
      return;
    }
    const scope = target.closest<HTMLElement>('[data-exp-scope]');
    if (scope) {
      const value = scope.dataset.expScope as string;
      if (value === '__custom') {
        const details = root.querySelector<HTMLDetailsElement>('.exp-advanced');
        if (details) {
          details.open = true;
          details.querySelector('input')?.focus();
        }
        return;
      }
      s.form.only = value;
      s.startError = '';
      const panel = el('expStart');
      if (panel) {
        renderStart(panel);
        hydrateIcons(panel);
        updateStartHint();
      }
      return;
    }
    const go = target.closest<HTMLElement>('[data-exp-go]');
    if (go && !(go as HTMLButtonElement).disabled) navigate(go.dataset.expGo as string);
  });

  root.addEventListener('keydown', (event) => {
    const target = event.target as HTMLElement;
    if (event.key === 'Enter' && target.matches('tr[data-exp-go]')) navigate(target.dataset.expGo as string);
  });

  root.addEventListener('input', (event) => {
    const target = event.target as HTMLInputElement;
    if (target.id === 'expName') {
      s.form.name = target.value;
      s.form.nameTouched = true;
    } else if (target.id === 'expWorkers') {
      s.form.workers = target.valueAsNumber;
    } else if (target.id === 'expOnly') {
      s.form.only = target.value;
    } else {
      return;
    }
    s.startError = '';
    syncScopeChips();
    updateStartHint();
  });

  root.addEventListener('change', (event) => {
    const target = event.target as HTMLSelectElement;
    if (target.id !== 'expSpec') return;
    const spec = s.specs?.find((item) => item.file === target.value);
    if (!spec) return;
    pickSpec(spec);
    s.form.only = '';
    s.startError = '';
    const panel = el('expStart');
    if (panel) {
      renderStart(panel);
      hydrateIcons(panel);
      updateStartHint();
    }
  });

  // A log inside a closed <details> has no scroll position: pin it to the bottom when opened.
  root.addEventListener('toggle', (event) => {
    const details = event.target;
    if (details instanceof HTMLDetailsElement && details.open) {
      details.querySelectorAll<HTMLElement>('.exp-pre').forEach((pre) => {
        pre.scrollTop = pre.scrollHeight;
      });
    }
  }, true);

  document.addEventListener('visibilitychange', () => {
    if (s.active && !document.hidden) requestTick();
  });

  window.addEventListener('popstate', () => {
    if (s.active && window.location.pathname.startsWith('/experiments')) syncFromLocation();
  });

  return { setActive };
}
