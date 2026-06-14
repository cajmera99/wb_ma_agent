/**
 * Subscribes to the SSE stream for a run and renders a live event log.
 * The parent passes streamUrl; this component manages its own EventSource lifecycle.
 */
import { useEffect, useRef, useState } from 'react'

const NODE_LABELS = {
  score_and_rank: 'Scoring acquirers',
  evaluate_coverage: 'Evaluating coverage',
  expand_candidate_pool: 'Expanding candidate pool',
  llm_rerank: 'LLM re-ranking',
  generate_rationales: 'Generating rationales',
  quality_gate: 'Quality gate',
  targeted_regeneration: 'Targeted regeneration',
}

const EVENT_ICONS = {
  'run.started': '▶',
  'node.started': '→',
  'node.completed': '✓',
  'node.error': '✗',
  'routing.decision': '⟶',
  'tool.called': '⚙',
  'tool.result': '↩',
  'rationale.generated': '✍',
  'validation.failed': '⚠',
  'validation.repaired': '✓✓',
  'llm.tokens_used': '◈',
  'run.completed': '🎉',
  'run.failed': '✗✗',
}

const S = {
  card: {
    background: '#fff',
    borderRadius: 8,
    padding: '20px 24px',
    boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
    marginBottom: 24,
  },
  title: { fontSize: 16, fontWeight: 700, color: '#003087', marginBottom: 12 },
  log: {
    background: '#0d1117',
    borderRadius: 6,
    padding: '12px 14px',
    maxHeight: 360,
    overflowY: 'auto',
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#c9d1d9',
  },
  event: { display: 'flex', gap: 8, marginBottom: 4, lineHeight: '18px' },
  time: { color: '#6e7681', minWidth: 56 },
  icon: { minWidth: 16 },
  nodeLabel: { color: '#58a6ff' },
  completed: { color: '#3fb950' },
  error: { color: '#f85149' },
  routing: { color: '#d2a8ff' },
  tool: { color: '#ffa657' },
  rationale: { color: '#79c0ff' },
}

function eventStyle(type) {
  if (type === 'run.completed' || type === 'node.completed' || type === 'validation.repaired') return S.completed
  if (type === 'run.failed' || type === 'node.error' || type === 'validation.failed') return S.error
  if (type === 'routing.decision') return S.routing
  if (type.startsWith('tool.')) return S.tool
  if (type === 'rationale.generated') return S.rationale
  return {}
}

function formatEvent(evt) {
  const type = evt.event_type
  const node = evt.node ? `[${NODE_LABELS[evt.node] || evt.node}] ` : ''
  const d = evt.data || {}

  if (type === 'node.completed' && d.total_acquirers_scored) return `${node}${d.total_acquirers_scored} acquirers scored — top: ${d.top_acquirer} (${d.top_score?.toFixed(1)})`
  if (type === 'routing.decision' && evt.node === 'quality_gate') {
    if (d.routing_to === 'regenerate_weak') {
      return `Quality gate → regenerating: ${(d.weak_acquirers || []).join(', ')}`
    }
    return `Quality gate → proceed to PDF (${d.reasoning || 'quality acceptable'})`
  }
  if (type === 'routing.decision') return `Coverage: ${d.candidates_above_threshold} above threshold → ${d.routing_to}`
  if (type === 'tool.called') return `${node}→ ${d.tool}(${JSON.stringify(d.args || {}).slice(0, 80)})`
  if (type === 'rationale.generated') return `${node}#${d.rank} ${d.acquirer} — ${d.conviction} conviction`
  if (type === 'node.completed' && node.includes('rerank')) return `${node}Final 10: ${(d.final_acquirers || []).join(', ')}`
  if (type === 'node.completed' && node.includes('rationale')) return `${node}${d.rationales_generated} generated, ${d.rationales_failed} failed`
  if (type === 'validation.failed') {
    const reason = d.error === 'forbidden_ebitda_attribution_detected'
      ? 'content violation detected — re-generating'
      : `schema error — re-generating (${(d.error || '').slice(0, 50)})`
    return `${node}${d.acquirer ? d.acquirer + ': ' : ''}${reason}`
  }
  if (type === 'validation.repaired') return `${node}${d.acquirer ? d.acquirer + ': ' : ''}corrected and re-generated ✓`
  if (type === 'run.completed') return `Run complete — PDF ready`
  if (type === 'run.failed') return `Run failed: ${d.error}`
  if (type === 'llm.tokens_used') return `${node}tokens: in=${d.input_tokens} out=${d.output_tokens}`
  return `${node}${type.split('.')[1]}`
}


export default function RunProgress({ runId, streamUrl, onComplete }) {
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState('connecting')
  const logRef = useRef(null)

  useEffect(() => {
    if (!streamUrl) return
    const es = new EventSource(streamUrl)

    es.onopen = () => setStatus('running')

    es.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data)
        // Ignore keepalive pings
        if (evt.event_type === 'keepalive') return
        setEvents((prev) => [...prev, evt])
        if (evt.event_type === 'run.completed') {
          setStatus('completed')
          onComplete && onComplete(evt.data)
          es.close()
        }
        if (evt.event_type === 'run.failed') {
          setStatus('failed')
          es.close()
        }
      } catch {}
    }

    es.onerror = () => {
      // onerror fires on normal server-close too — only mark error if not already done
      setStatus((s) => (s === 'completed' || s === 'failed' ? s : 'error'))
      es.close()
    }

    return () => es.close()
  }, [streamUrl])

  // Auto-scroll the log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [events])

  const statusColor = { running: '#58a6ff', completed: '#3fb950', failed: '#f85149', connecting: '#888', error: '#f85149' }

  return (
    <div style={S.card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={S.title}>Agent Progress</div>
        <span style={{ fontSize: 12, fontWeight: 600, color: statusColor[status] || '#888' }}>
          {status.toUpperCase()}
        </span>
      </div>
      <div style={S.log} ref={logRef}>
        {events.length === 0 && (
          <div style={{ color: '#6e7681' }}>Waiting for events…</div>
        )}
        {events.map((evt, i) => {
          const ts = new Date(evt.timestamp).toLocaleTimeString('en-US', { hour12: false, timeStyle: 'medium' })
          return (
            <div key={i} style={{ ...S.event, ...eventStyle(evt.event_type) }}>
              <span style={S.time}>{ts}</span>
              <span style={S.icon}>{EVENT_ICONS[evt.event_type] || '·'}</span>
              <span>{formatEvent(evt)}</span>
            </div>
          )
        })}
        {status === 'running' && (
          <div style={{ color: '#58a6ff', marginTop: 4 }}>● processing…</div>
        )}
      </div>
    </div>
  )
}
