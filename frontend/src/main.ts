import './styles.css';
import { fetchJson, postJson } from './api';
import DOMPurify from 'dompurify';
import {
  Activity,
  AlertTriangle,
  Archive,
  Bot,
  Boxes,
  Check,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Gauge,
  GitBranch,
  Info,
  Loader2,
  Play,
  RefreshCw,
  Send,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  TerminalSquare,
  X,
  XCircle,
} from 'lucide';
import type { IconNode } from 'lucide';
import { marked } from 'marked';
import type { Bootstrap, JsonMap } from './types';
import { createExperiments } from './experiments';

type View = 'experiments' | 'settings';

const state = {
  view: (window.location.pathname === '/settings' ? 'settings' : 'experiments') as View,
  bootstrap: null as Bootstrap | null,
  bootstrapError: '',
  expandedToolPartIds: new Set<string>(),
  renderSignatures: new Map<string, string>(),
  settings: {
    mainAuthMode: 'manual',
    mainModel: '',
    mainBaseUrl: '',
    mainApiKey: '',
  },
};

if (window.location.pathname !== '/settings' && !isExperimentsPath(window.location.pathname)) {
  history.replaceState({}, '', '/experiments');
}

marked.setOptions({
  breaks: true,
  gfm: true,
});

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('Missing #app root');

app.innerHTML = `
  <section id="experimentsView" class="experiments-view" hidden></section>

  <section id="settingsView" class="settings-view" hidden>
    <header class="settings-header">
      <div class="settings-header-inner">
        <div class="settings-header-text">
          <div class="settings-eyebrow"><span data-icon="SlidersHorizontal"></span><span>工作区</span></div>
          <h1 class="settings-title">设置</h1>
          <p class="settings-subtitle">工作区信息和主模型配置。模型设置对之后启动的实验生效。</p>
        </div>
        <div class="settings-header-actions">
          <button id="settingsDoneBtn" type="button" class="settings-done-btn"><span data-icon="ChevronLeft"></span><span>返回实验</span></button>
        </div>
      </div>
    </header>
    <div id="settingsContent" class="settings-content">
      <div id="settingsWorkspaceSection"></div>
      <div id="settingsFormSection"></div>
    </div>
  </section>

  <dialog id="detailDialog">
    <form method="dialog">
      <header>
        <h3 id="dialogTitle">详情</h3>
        <div class="dialog-actions">
          <button value="close" class="icon-btn ghost" type="submit" title="关闭"><span data-icon="X"></span></button>
        </div>
      </header>
      <pre id="dialogBody"></pre>
    </form>
  </dialog>
`;

const iconMap: Record<string, IconNode> = {
  Activity,
  AlertTriangle,
  Archive,
  Bot,
  Boxes,
  Check,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Gauge,
  GitBranch,
  Info,
  Loader2,
  Play,
  RefreshCw,
  Send,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  TerminalSquare,
  X,
  XCircle,
};

const els = {
  experimentsView: query<HTMLElement>('#experimentsView'),
  settingsView: query<HTMLElement>('#settingsView'),
  settingsContent: query<HTMLElement>('#settingsContent'),
  settingsWorkspaceSection: query<HTMLElement>('#settingsWorkspaceSection'),
  settingsFormSection: query<HTMLElement>('#settingsFormSection'),
  settingsDoneBtn: query<HTMLButtonElement>('#settingsDoneBtn'),
  dialog: query<HTMLDialogElement>('#detailDialog'),
  dialogTitle: query('#dialogTitle'),
  dialogBody: query('#dialogBody'),
};

hydrateIcons();

const experiments = createExperiments({
  root: els.experimentsView,
  escapeHtml,
  hydrateIcons,
  renderStable,
  runAction: runQuietAction,
  emptyState,
  messageHtml,
  normalizeChatParts,
  bindToolCards,
  goSettings: openSettingsView,
});

els.settingsDoneBtn.addEventListener('click', () => navigate('experiments'));

window.addEventListener('popstate', () => {
  state.view = window.location.pathname === '/settings' ? 'settings' : 'experiments';
  if (state.view === 'experiments' && !isExperimentsPath(window.location.pathname)) {
    history.replaceState({}, '', '/experiments');
  }
  render();
  if (state.view === 'settings') syncAndRenderSettingsForm();
});

function navigate(view: View) {
  state.view = view;
  const path = view === 'settings' ? '/settings' : '/experiments';
  if (window.location.pathname !== path) history.pushState({}, '', path);
  render();
  if (view === 'settings') syncAndRenderSettingsForm();
}

function openSettingsView() {
  navigate('settings');
}

function isExperimentsPath(path: string) {
  return path === '/experiments' || path.startsWith('/experiments/');
}

function render() {
  const experimentsMode = state.view === 'experiments';
  els.experimentsView.hidden = !experimentsMode;
  els.settingsView.hidden = experimentsMode;
  document.body.classList.toggle('experiments-page', experimentsMode);
  document.body.classList.toggle('settings-page', !experimentsMode);
  experiments.setActive(experimentsMode);
  if (!experimentsMode) renderSettings();
}

// ------------------------------------------------------------ bootstrap

async function loadBootstrap() {
  try {
    const data = await fetchJson<Bootstrap>('/api/bootstrap');
    state.bootstrapError = '';
    applyBootstrapSnapshot(data);
  } catch (error) {
    state.bootstrapError = errText(error);
    render();
  }
}

function applyBootstrapSnapshot(data: Bootstrap) {
  state.bootstrap = data;
  syncSettingsFromBootstrap(data);
  renderSettingsForm();
  render();
}

function syncSettingsFromBootstrap(data: Bootstrap) {
  const main = data.llmConfig?.config || {};
  state.settings.mainAuthMode = main.authMode || 'manual';
  state.settings.mainModel = main.model || '';
  state.settings.mainBaseUrl = main.baseUrl || '';
  // apiKey is masked (e.g. "****1234"); the editor starts empty on each seed.
  state.settings.mainApiKey = '';
}

function syncAndRenderSettingsForm() {
  if (!state.bootstrap) return;
  syncSettingsFromBootstrap(state.bootstrap);
  renderSettingsForm();
}

// ------------------------------------------------------------ settings view

function renderSettings() {
  const data = state.bootstrap;
  if (!data) {
    els.settingsWorkspaceSection.innerHTML = state.bootstrapError
      ? emptyState(`无法读取工作区：${state.bootstrapError}`, 'AlertTriangle')
      : emptyState('正在读取工作区…', 'Loader2');
    hydrateIcons(els.settingsWorkspaceSection);
    return;
  }
  renderStable(
    'settings-workspace',
    [data.state, data.dryRun, data.debugEnabled, data.runtime?.workspaceUv?.workspace],
    () => renderSettingsWorkspace(data),
  );
}

function renderSettingsWorkspace(data: Bootstrap) {
  const workspaceRows = [
    {
      label: '工作区路径',
      value: `<span class="settings-readout-mono">${escapeHtml(data.runtime?.workspaceUv?.workspace || '')}</span>`,
      copyable: data.runtime?.workspaceUv?.workspace || '',
    },
  ];

  els.settingsWorkspaceSection.innerHTML = `
    <div class="settings-section">
      <div class="settings-section-header">工作区</div>
      <div class="settings-card">
        ${workspaceRows.map((row) => `
          <div class="settings-row">
            <div class="settings-row-label">${escapeHtml(row.label)}</div>
            <div class="settings-row-control">${row.value}${row.copyable ? `<button class="settings-icon-btn" type="button" data-copy="${escapeHtml(row.copyable)}" title="复制"><span data-icon="Archive"></span></button>` : ''}</div>
          </div>
        `).join('')}
        <div class="settings-row">
          <div class="settings-row-label">详细信息</div>
          <div class="settings-row-control settings-inline-actions">
            <button class="settings-secondary-btn" type="button" data-settings-detail="state"><span data-icon="Archive"></span><span>查看状态文件</span></button>
            <button class="settings-secondary-btn" type="button" data-settings-detail="llm"><span data-icon="Settings2"></span><span>查看模型配置</span></button>
          </div>
        </div>
      </div>
      <p class="settings-section-foot">工作区路径由服务启动参数 <code>TS_HARNESS_WORKSPACE</code> 决定。如需修改，请重启服务。</p>
    </div>
  `;
  bindSettingsWorkspaceHandlers(els.settingsWorkspaceSection);
  hydrateIcons(els.settingsWorkspaceSection);
}

function renderSettingsForm() {
  els.settingsFormSection.innerHTML = renderMainLlmSection();
  bindSettingsFormHandlers(els.settingsFormSection);
  hydrateIcons(els.settingsFormSection);
}

function renderMainLlmSection(): string {
  const s = state.settings;
  const authOptions = [
    { value: 'manual', label: '手动配置' },
    { value: 'sdk-default', label: 'SDK 默认' },
  ];
  return `
    <div class="settings-section">
      <div class="settings-section-header">主模型</div>
      <div class="settings-card">
        <div class="settings-row">
          <div class="settings-row-label">认证方式</div>
          <div class="settings-row-control">${segmentedHtml('mainAuthMode', authOptions.map((o) => o.value), s.mainAuthMode, authOptions.map((o) => o.label))}</div>
        </div>
        <div class="settings-row">
          <div class="settings-row-label">模型</div>
          <div class="settings-row-control"><input class="settings-input" type="text" id="mainModel" value="${escapeHtml(s.mainModel)}" placeholder="例如 deepseek-v4-pro" /></div>
        </div>
        <div class="settings-row">
          <div class="settings-row-label">接口地址</div>
          <div class="settings-row-control"><input class="settings-input" type="text" id="mainBaseUrl" value="${escapeHtml(s.mainBaseUrl)}" placeholder="https://api.deepseek.com" /></div>
        </div>
        <div class="settings-row">
          <div class="settings-row-label">API 密钥</div>
          <div class="settings-row-control"><input class="settings-input" type="password" id="mainApiKey" value="" placeholder="留空表示保持当前密钥" autocomplete="off" /></div>
        </div>
        <div class="settings-card-footer">
          <button id="saveMainLlmBtn" type="button" class="settings-primary-btn"><span data-icon="Check"></span><span>保存主模型</span></button>
          <button id="resetMainLlmBtn" type="button" class="settings-secondary-btn"><span data-icon="RefreshCw"></span><span>恢复文件默认值</span></button>
          <span class="settings-save-status" id="mainSaveStatus"></span>
        </div>
      </div>
      <p class="settings-section-foot">供实验智能体使用。保存到 <code>&lt;workspace&gt;/config.llm.json</code>，新启动的实验单元格会读取它；环境变量 <code>TS_HARNESS_LLM_*</code> 优先。</p>
    </div>
  `;
}

function segmentedHtml(group: string, values: string[], current: string, labels: string[] | null = null): string {
  return `
    <div class="settings-segmented" data-segmented-group="${escapeHtml(group)}" role="group">
      ${values.map((v, i) => {
        const label = labels ? labels[i] : v;
        const selected = (current || '') === v ? 'selected' : '';
        return `<button type="button" class="settings-segmented-item ${selected}" data-value="${escapeHtml(v)}">${escapeHtml(label)}</button>`;
      }).join('')}
    </div>
  `;
}

function bindSettingsFormHandlers(root: HTMLElement) {
  for (const group of root.querySelectorAll<HTMLElement>('[data-segmented-group]')) {
    const name = group.dataset.segmentedGroup || '';
    for (const item of group.querySelectorAll<HTMLButtonElement>('.settings-segmented-item')) {
      item.addEventListener('click', () => {
        const value = item.dataset.value || '';
        for (const sibling of group.querySelectorAll<HTMLButtonElement>('.settings-segmented-item')) {
          sibling.classList.toggle('selected', sibling === item);
        }
        if (name === 'mainAuthMode') state.settings.mainAuthMode = value;
      });
    }
  }
  for (const input of root.querySelectorAll<HTMLInputElement>('.settings-input')) {
    input.addEventListener('input', () => {
      const id = input.id;
      if (id === 'mainModel') state.settings.mainModel = input.value;
      else if (id === 'mainBaseUrl') state.settings.mainBaseUrl = input.value;
      else if (id === 'mainApiKey') state.settings.mainApiKey = input.value;
    });
  }
  root.querySelector<HTMLButtonElement>('#saveMainLlmBtn')?.addEventListener('click', () => saveLlmConfig());
  root.querySelector<HTMLButtonElement>('#resetMainLlmBtn')?.addEventListener('click', () => resetMainLlmConfig());
}

function bindSettingsWorkspaceHandlers(root: HTMLElement) {
  for (const detailBtn of root.querySelectorAll<HTMLButtonElement>('[data-settings-detail]')) {
    detailBtn.addEventListener('click', () => {
      if (detailBtn.dataset.settingsDetail === 'state') {
        showDetail('工作区状态文件', state.bootstrap?.state);
      } else {
        showDetail('模型配置', state.bootstrap?.llmConfig);
      }
    });
  }
  for (const copyBtn of root.querySelectorAll<HTMLButtonElement>('[data-copy]')) {
    copyBtn.addEventListener('click', async () => {
      const text = copyBtn.dataset.copy || '';
      try {
        await navigator.clipboard.writeText(text);
        const original = copyBtn.innerHTML;
        copyBtn.innerHTML = '<span data-icon="Check"></span>';
        copyBtn.classList.add('copied');
        hydrateIcons(copyBtn);
        window.setTimeout(() => {
          copyBtn.innerHTML = original;
          copyBtn.classList.remove('copied');
          hydrateIcons(copyBtn);
        }, 1100);
      } catch (error) {
        showDetail('复制失败', { message: errText(error) });
      }
    });
  }
}

async function saveLlmConfig() {
  const s = state.settings;
  const body: JsonMap = {
    authMode: s.mainAuthMode,
    model: s.mainModel,
    apiKey: s.mainApiKey || undefined,
    baseUrl: s.mainBaseUrl,
  };
  try {
    const result = await postJson('/api/llm-config', body);
    state.settings.mainApiKey = '';
    applyBootstrapSnapshot(result.bootstrap);
    const status = els.settingsContent.querySelector<HTMLElement>('#mainSaveStatus');
    if (status) {
      status.textContent = '已保存';
      status.classList.add('saved');
      window.setTimeout(() => {
        status.textContent = '';
        status.classList.remove('saved');
      }, 1800);
    }
  } catch (error) {
    showDetail('保存主模型失败', { message: errText(error) });
  }
}

async function resetMainLlmConfig() {
  if (!confirm('是否将主模型恢复为文件默认值？这会清空保存在 <workspace>/config.llm.json 中的模型和接口地址设置；API 密钥会保留。')) return;
  try {
    const result = await postJson('/api/llm-config', {
      authMode: 'sdk-default',
      model: '',
      baseUrl: '',
    });
    applyBootstrapSnapshot(result.bootstrap);
  } catch (error) {
    showDetail('恢复主模型失败', { message: errText(error) });
  }
}

// ------------------------------------------------------------ agent parts

function messageHtml(part: JsonMap) {
  const role = part.role || 'system';
  const roleClass = ['user', 'assistant', 'system', 'tool'].includes(role) ? role : 'system';
  const text = part.displayText || part.text || '';
  const source = part.sourceLabel ? `${sourceLabel(part.sourceLabel)} · ` : '';
  if (part.type === 'loading') {
    return `
      <article class="message assistant loading">
        <div class="message-role"><span data-icon="Loader2"></span>${escapeHtml(source)}助手 · 运行中 · ${formatTime(part.timestamp)}</div>
        <div class="message-text loading-line"><span>${escapeHtml(part.text)}</span></div>
      </article>
    `;
  }
  if (part.type === 'tool_call' || part.type === 'tool_result') {
    const summary = toolSummary(part);
    const detail = toolDetail(part);
    const toolName = toolNameForPart(part);
    const failed = part.status === 'failed';
    const done = !failed && (part.type === 'tool_result' || part.status === 'completed');
    const partId = String(part.id || `${part.timestamp || ''}:${toolName}:${part.type || ''}`);
    const expanded = state.expandedToolPartIds.has(partId);
    return `
      <article class="message ${escapeHtml(roleClass)} tool-collapsed${expanded ? ' expanded' : ''}" data-part-id="${escapeHtml(partId)}">
        <button class="tool-card${expanded ? ' expanded' : ''}" type="button" data-tool-toggle>
          <span class="tool-status ${failed ? 'failed' : done ? 'done' : 'running'}"><span data-icon="${failed ? 'XCircle' : done ? 'Check' : 'TerminalSquare'}"></span></span>
          <span class="tool-copy">
            <span class="tool-title">${escapeHtml(summary)}</span>
            <span class="tool-meta">${escapeHtml(source)}${escapeHtml(toolName)} · ${escapeHtml(partTypeLabel(part.type || 'tool'))} · ${formatTime(part.timestamp)}</span>
          </span>
          <span class="tool-chevron" data-icon="ChevronRight"></span>
        </button>
        <pre class="tool-detail" ${expanded ? '' : 'hidden'}>${escapeHtml(detail)}</pre>
      </article>
    `;
  }
  return `
    <article class="message ${escapeHtml(roleClass)}">
      <div class="message-role"><span data-icon="${role === 'user' ? 'Send' : role === 'assistant' ? 'Bot' : 'Info'}"></span>${escapeHtml(source)}${escapeHtml(roleLabel(role))} · ${escapeHtml(partTypeLabel(part.type || 'text'))} · ${formatTime(part.timestamp)}</div>
      <div class="message-text markdown-body">${renderMarkdown(text)}</div>
    </article>
  `;
}

function toolSummary(part: JsonMap) {
  if (part.type === 'tool_call') {
    const intend = typeof part.intend === 'string' && part.intend.trim() ? part.intend.trim() : '';
    return intend || `调用工具：${toolNameForPart(part)}`;
  }
  const text = part.resultText || part.displayText || part.text || '';
  const firstLine = text.split('\n').find((line: string) => line.trim()) || '工具结果';
  return firstLine.length > 120 ? `${firstLine.slice(0, 117)}...` : firstLine;
}

function toolDetail(part: JsonMap) {
  if (part.type === 'tool_call') {
    const chunks = [
      `工具：${toolNameForPart(part)}`,
      `状态：${statusLabel(part.status || 'pending')}`,
      '',
      '调用意图:',
      part.intend || '',
      '',
      '调用参数:',
      JSON.stringify(part.input ?? {}, null, 2),
    ];
    if (part.status === 'completed' || part.status === 'failed' || part.resultText) {
      chunks.push('', '调用结果:', part.resultText || '');
    } else {
      chunks.push('', '调用结果:', '等待工具返回结果。');
    }
    return chunks.join('\n');
  }
  return part.resultText || part.displayText || part.text || '';
}

function toolNameForPart(part: JsonMap) {
  return part.name || 'tool';
}

function normalizeChatParts(parts: JsonMap[]): JsonMap[] {
  return parts
    .map((part: JsonMap): JsonMap | null => {
      const failureText = protocolFailureText(part);
      if (failureText) {
        return {
          ...part,
          role: 'system',
          type: 'text',
          text: failureText,
          displayText: failureText,
        };
      }
      if (!isDisplayableSessionPart(part)) return null;
      const displayText = displayTextForPart(part);
      return displayText.trim() ? { ...part, displayText } : null;
    })
    .filter((part: JsonMap | null): part is JsonMap => part !== null);
}

function isDisplayableSessionPart(part: JsonMap): boolean {
  if (part.type === 'loading') return true;
  if (part.type === 'tool_call' || part.type === 'tool_result') return true;
  // A successful turn result carries no user-facing content and is hidden;
  // a failed one is shown as a system message (see protocolFailureText).
  if (part.role === 'system' && part.type === 'result') return part.status === 'failed';
  return true;
}

function protocolFailureText(part: JsonMap): string {
  if (part.role !== 'system' || part.type !== 'result' || part.status !== 'failed') return '';
  const detail = String(part.resultText || part.text || '').trim();
  return detail ? `执行失败：${detail}` : '执行失败。';
}

function displayTextForPart(part: JsonMap) {
  if (part.type === 'loading') return part.text || '运行中';
  if (part.type === 'tool_call') {
    return part.intend || `调用工具：${toolNameForPart(part)}`;
  }
  if (part.type === 'tool_result') {
    const result = part.resultText || part.text || '';
    // An empty result is transported as the placeholder "Tool result".
    const text = result === 'Tool result' ? '' : result;
    return text || (part.status === 'failed' ? '工具执行失败' : '');
  }
  if (part.role === 'system' && part.type === 'result') return '';
  return part.text || '';
}

function bindToolCards(root: HTMLElement) {
  for (const button of root.querySelectorAll<HTMLButtonElement>('[data-tool-toggle]')) {
    button.addEventListener('click', () => {
      const detail = button.nextElementSibling as HTMLElement | null;
      if (!detail) return;
      const expanded = detail.hidden === true;
      detail.hidden = !expanded;
      const article = button.closest<HTMLElement>('.tool-collapsed');
      const partId = article?.dataset.partId;
      if (partId) {
        if (expanded) state.expandedToolPartIds.add(partId);
        else state.expandedToolPartIds.delete(partId);
      }
      article?.classList.toggle('expanded', expanded);
      button.classList.toggle('expanded', expanded);
      if (expanded) requestAnimationFrame(() => article?.scrollIntoView({ block: 'nearest' }));
    });
  }
}

function roleLabel(role: any) {
  const value = String(role || '');
  if (value === 'user') return '用户';
  if (value === 'assistant') return '助手';
  if (value === 'system') return '系统';
  return value;
}

function partTypeLabel(type: any) {
  const value = String(type || '');
  if (value === 'text') return '文本';
  if (value === 'tool_call') return '工具调用';
  if (value === 'tool_use') return '工具使用';
  if (value === 'tool_result') return '工具结果';
  if (value === 'tool') return '工具';
  if (value === 'loading') return '运行中';
  if (value === 'raw') return '原始事件';
  if (value === 'result') return '结果';
  return value;
}

function sourceLabel(source: any) {
  const value = String(source || '');
  if (value === 'main') return '主会话';
  if (value === 'builder') return '构建器';
  if (value === 'harness') return '系统';
  return value;
}

function statusLabel(status: any) {
  const value = String(status || 'idle');
  if (value === 'done' || value === 'completed' || value === 'ready') return '已完成';
  if (value === 'active' || value === 'running' || value === 'building') return '运行中';
  if (value === 'failed') return '失败';
  if (value === 'paused') return '已暂停';
  if (value === 'waiting_approval') return '等待确认';
  if (value === 'pending') return '待执行';
  if (value === 'idle') return '空闲';
  return value;
}

// ------------------------------------------------------------ shared helpers

function emptyState(text: string, icon = 'Info') {
  return `<div class="empty"><span data-icon="${icon}"></span><span>${escapeHtml(text)}</span></div>`;
}

function renderMarkdown(text: string) {
  const html = marked.parse(text || '', { async: false }) as string;
  return DOMPurify.sanitize(html);
}

async function runQuietAction(fn: () => Promise<void>) {
  try {
    await fn();
  } catch (error) {
    showDetail('错误', { message: errText(error) });
  }
}

function showDetail(title: string, value: any) {
  els.dialogTitle.textContent = title;
  const text = typeof value === 'string' ? value : JSON.stringify(value ?? null, null, 2);
  els.dialogBody.textContent = text || '（空）';
  els.dialog.showModal();
}

function renderStable(key: string, value: any, renderFn: () => void) {
  const signature = JSON.stringify(value ?? null);
  if (state.renderSignatures.get(key) === signature) return;
  state.renderSignatures.set(key, signature);
  renderFn();
}

function formatTime(value: string) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function errText(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function escapeHtml(value: any) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function query<T extends Element = HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}

function hydrateIcons(root: ParentNode = document) {
  for (const holder of root.querySelectorAll<HTMLElement>('[data-icon]')) {
    if (holder.dataset.iconHydrated === 'true') continue;
    const name = holder.dataset.icon || '';
    const iconNode = iconMap[name];
    if (!iconNode) continue;
    const svg = createIcon(iconNode);
    holder.replaceChildren(svg);
    holder.dataset.iconHydrated = 'true';
  }
}

function createIcon(iconNode: IconNode) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  for (const [tag, attrs] of iconNode) {
    const child = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, String(value));
    svg.appendChild(child);
  }
  return svg;
}

render();
void loadBootstrap();
