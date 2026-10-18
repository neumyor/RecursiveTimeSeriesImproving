/**
 * Apply the Python runner's role/variant tool contract to every SDK-created
 * agent.  Cordis restrictions are scoped, so an in-process subagent retains
 * its own local report channel while inherited global tools stay constrained.
 */
export const name = 'harnessing-ts-tool-allowlist'
export const inject = ['agents', 'tools']

const SUBAGENT_ONLY_TOOLS = new Set([
  'subagent',
  'subagent_fork',
  'send_message',
  'interrupt_agent',
  'list_agents',
])

function configuredTools() {
  return new Set(
    (process.env.TS_HARNESS_DSH_ALLOWED_TOOLS || '')
      .split(',')
      .map((name) => name.trim())
      .filter(Boolean),
  )
}

function isDelegatedAgent(agent) {
  return agent.session?.header?.origin === 'subagent'
    || Number(agent.options?.subagentDepth || 0) > 0
}

export function apply(ctx) {
  const allowed = configuredTools()
  // `agent/session-start` is the first injection point where the scoped
  // agent context and its effect-backed tool restrictions exist.
  ctx.on('agent/session-start', ({ agent }) => {
    const visible = agent.ctx.tools.schemas().map((schema) => schema.name)
    const delegated = isDelegatedAgent(agent)
    const permitted = visible.filter((name) => {
      if (!allowed.has(name)) return false
      // A delegated worker may investigate, edit, run commands, and use an
      // explicitly enabled web provider, but it must not advance the harness
      // protocol or recursively delegate. The parent owns completion.
      if (delegated && (name.startsWith('mcp__ts_harness__') || SUBAGENT_ONLY_TOOLS.has(name))) return false
      return true
    })
    // `tools.restrict()` intentionally rejects an empty allow-list.  Denying
    // the discovered global surface gives reasoner-only sessions zero native
    // tools while still leaving any child-scoped reporting tool intact.
    if (permitted.length > 0) {
      agent.ctx.tools.restrict({ allow: permitted })
    } else if (visible.length > 0) {
      agent.ctx.tools.restrict({ deny: visible })
    }
  })
}
