import { useState, useEffect } from 'react'
import TargetForm from './components/TargetForm'
import RunProgress from './components/RunProgress'
import AcquirerCard from './components/AcquirerCard'
import RunHistory from './components/RunHistory'

const S = {
  root: { minHeight: '100vh', background: '#f0f4f8' },
  header: {
    background: '#003087',
    color: '#fff',
    padding: '14px 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
  },
  headerTitle: { fontSize: 18, fontWeight: 700, letterSpacing: 0.3 },
  headerSub: { fontSize: 11, color: '#c8d8ee', marginTop: 2 },
  layout: {
    display: 'grid',
    gridTemplateColumns: '220px 1fr',
    gap: 24,
    padding: '24px 32px',
    maxWidth: 1200,
    margin: '0 auto',
  },
  main: {},
  resultsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  resultsTitle: { fontSize: 16, fontWeight: 700, color: '#003087' },
  pdfBtn: {
    padding: '7px 16px', background: '#003087', color: '#fff',
    border: 'none', borderRadius: 5, fontSize: 12, fontWeight: 600,
    cursor: 'pointer', textDecoration: 'none', display: 'inline-block',
  },
}

export default function App() {
  const [runId, setRunId] = useState(null)
  const [streamUrl, setStreamUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [historicalTarget, setHistoricalTarget] = useState(null)

  // On mount: restore the most recent completed run from the backend so a
  // browser refresh doesn't wipe the main content (the server still has it).
  useEffect(() => {
    fetch('/api/runs')
      .then((r) => (r.ok ? r.json() : []))
      .then((runs) => {
        const latest = runs.find((r) => r.status === 'completed')
        if (!latest) return
        fetch(`/api/runs/${latest.run_id}/result`)
          .then((r) => (r.ok ? r.json() : null))
          .then((data) => {
            if (!data) return
            setRunId(latest.run_id)
            setResult(data)
            setHistoricalTarget(latest.target || null)
            setRefreshKey((k) => k + 1)
          })
      })
      .catch(() => {})
  }, [])

  const handleRunStarted = (id, url) => {
    setRunId(id)
    setStreamUrl(url)
    setLoading(true)
    setResult(null)
    setHistoricalTarget(null)
    setRefreshKey((k) => k + 1)
  }

  const handleComplete = (data) => {
    setLoading(false)
    setResult(data)
    setRefreshKey((k) => k + 1)
  }

  const handleSelectHistoricalRun = (id, data, target, isRunning = false) => {
    setRunId(id)
    setResult(isRunning ? null : data)
    setHistoricalTarget(isRunning ? null : (target || null))
    setLoading(isRunning)
    // Reconnect to the SSE stream so the progress panel reappears
    setStreamUrl(isRunning ? `/api/runs/${id}/stream` : null)
  }

  const handleNewAnalysis = () => {
    setHistoricalTarget(null)
    setResult(null)
    setRunId(null)
    setStreamUrl(null)
  }

  const rationales = result?.rationales
    ? [...result.rationales].sort((a, b) => (a.rank || 99) - (b.rank || 99))
    : []

  return (
    <div style={S.root}>
      <header style={S.header}>
        <div>
          <div style={S.headerTitle}>M&A Acquirer Identification Engine</div>
          <div style={S.headerSub}>William Blair — AI Innovation Assessment</div>
        </div>
        <div style={{ fontSize: 11, color: '#c8d8ee' }}>Powered by LangGraph + GPT-4o</div>
      </header>

      <div style={S.layout}>
        {/* Sidebar — run history */}
        <aside>
          <RunHistory onSelectRun={handleSelectHistoricalRun} activeRunId={runId} refreshKey={refreshKey} onNewAnalysis={handleNewAnalysis} />
        </aside>

        {/* Main content */}
        <main style={S.main}>
          <TargetForm
            onRunStarted={handleRunStarted}
            loading={loading}
            historicalTarget={historicalTarget}
            onNewAnalysis={handleNewAnalysis}
          />

          {/* API endpoint links — always visible */}
          <div style={{
            background: '#f7f9fc', border: '1px solid #e2e8f0', borderRadius: 6,
            padding: '8px 14px', marginBottom: 12, fontSize: 11,
            display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 4,
          }}>
            <span style={{ fontWeight: 600, color: '#003087', marginRight: 6 }}>
              System:
            </span>
            <a
              href="/api/graph"
              target="_blank"
              rel="noopener noreferrer"
              style={{ marginRight: 14, color: '#7c3aed', textDecoration: 'none', fontWeight: 600 }}
            >
              Agent Graph ↗
            </a>
            {runId && (
              <>
                <span style={{ color: '#cbd5e0', marginRight: 14 }}>|</span>
                <span style={{ fontWeight: 600, color: '#003087', marginRight: 6 }}>Run:</span>
                {[
                  { label: 'Live SSE stream', path: `/api/runs/${runId}/stream` },
                  { label: 'Full event log',  path: `/api/runs/${runId}/events` },
                  { label: 'Final result',    path: `/api/runs/${runId}/result` },
                ].map(({ label, path }) => (
                  <a
                    key={label}
                    href={path}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ marginRight: 14, color: '#2b6cb0', textDecoration: 'none', fontFamily: 'monospace' }}
                  >
                    {label} ↗
                  </a>
                ))}
              </>
            )}
          </div>

          {streamUrl && (
            <RunProgress
              runId={runId}
              streamUrl={streamUrl}
              onComplete={handleComplete}
            />
          )}

          {rationales.length > 0 && (
            <div>
              <div style={S.resultsHeader}>
                <div style={S.resultsTitle}>
                  Top {rationales.length} Acquirers
                </div>
                {runId && (
                  <a
                    href={`/api/runs/${runId}/pdf`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={S.pdfBtn}
                  >
                    Download PDF Report
                  </a>
                )}
              </div>
              {rationales.map((r) => (
                <AcquirerCard key={r.acquirer_name} rationale={r} />
              ))}
            </div>
          )}
        </main>
      </div>

      <footer style={{
        textAlign: 'center', padding: '10px 32px',
        fontSize: 10, color: '#a0aec0', letterSpacing: 0.3,
        borderTop: '1px solid #e2e8f0',
      }}>
        {`v${__APP_VERSION__} · built ${new Date(__BUILD_TIME__).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}`}
      </footer>
    </div>
  )
}
