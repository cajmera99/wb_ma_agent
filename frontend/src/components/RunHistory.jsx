/**
 * Sidebar panel showing previous runs fetched from GET /api/runs.
 * Clicking a completed run loads its result from GET /api/runs/{id}/result.
 */
import { useEffect, useState } from 'react'

const STATUS_COLORS = { completed: '#1E8449', running: '#003087', failed: '#C0392B' }

const S = {
  panel: {
    background: '#fff',
    borderRadius: 8,
    boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
    padding: '16px 18px',
    maxHeight: '85vh',
    overflowY: 'auto',
  },
  title: { fontSize: 14, fontWeight: 700, color: '#003087', marginBottom: 0 },
  row: (status) => ({
    padding: '10px 0',
    borderBottom: '1px solid #f0f4f8',
    cursor: status === 'failed' ? 'default' : 'pointer',
    opacity: status === 'failed' ? 0.5 : 1,
  }),
  rowInner: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  sector: { fontSize: 13, fontWeight: 600, color: '#1a202c' },
  ev: { fontSize: 11, color: '#718096', marginTop: 2 },
  status: (s) => ({ fontSize: 10, fontWeight: 700, color: STATUS_COLORS[s] || '#718096' }),
  time: { fontSize: 10, color: '#a0aec0', marginTop: 2 },
  empty: { fontSize: 12, color: '#a0aec0', textAlign: 'center', padding: '20px 0' },
}

export default function RunHistory({ onSelectRun, activeRunId, refreshKey, onNewAnalysis }) {
  const [runs, setRuns] = useState([])

  const load = async () => {
    try {
      const res = await fetch('/api/runs')
      if (res.ok) setRuns(await res.json())
    } catch {}
  }

  // Re-fetch whenever the parent signals a change (run started or completed)
  useEffect(() => { load() }, [refreshKey])

  const handleClick = async (run) => {
    if (run.status === 'running') {
      // Reconnect to the live SSE stream for this run
      onSelectRun(run.run_id, null, null, true)
      return
    }
    if (run.status !== 'completed') return
    try {
      const res = await fetch(`/api/runs/${run.run_id}/result`)
      if (res.ok) {
        const data = await res.json()
        onSelectRun(run.run_id, data, run.target, false)
      }
    } catch {}
  }

  return (
    <div style={S.panel}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={S.title}>Run History</div>
        <button
          onClick={onNewAnalysis}
          style={{
            padding: '5px 10px', background: '#003087', color: '#fff',
            border: 'none', borderRadius: 5, fontSize: 11, fontWeight: 600,
            cursor: 'pointer', whiteSpace: 'nowrap',
          }}
        >
          + New
        </button>
      </div>
      {runs.length === 0 && (
        <div style={S.empty}>No runs yet</div>
      )}
      {runs.map((r) => {
        const isActive = r.run_id === activeRunId
        const target = r.target || {}
        const ts = r.started_at ? new Date(r.started_at).toLocaleString() : ''
        return (
          <div
            key={r.run_id}
            style={{
              ...S.row(r.status),
              background: isActive ? '#EAF0F8' : 'transparent',
              borderRadius: isActive ? 5 : 0,
              paddingLeft: isActive ? 8 : 0,
            }}
            onClick={() => handleClick(r)}
          >
            <div style={S.rowInner}>
              <div style={S.sector}>{target.sector || 'Unknown sector'}</div>
              <span style={S.status(r.status)}>{r.status?.toUpperCase()}</span>
            </div>
            <div style={S.ev}>
              ${target.deal_size_mm}M &middot; {target.geography} &middot; {target.ownership}
            </div>
            <div style={S.time}>{ts}</div>
          </div>
        )
      })}
    </div>
  )
}
