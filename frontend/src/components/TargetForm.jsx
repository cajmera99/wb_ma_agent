import { useState, useEffect } from 'react'

const S = {
  card: {
    background: '#fff',
    borderRadius: 8,
    padding: '24px 28px',
    boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
    marginBottom: 24,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: { fontSize: 18, fontWeight: 700, color: '#003087' },
  historicalBadge: {
    fontSize: 11, fontWeight: 600, color: '#718096',
    background: '#EDF2F7', padding: '3px 10px', borderRadius: 10,
  },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px 20px' },
  label: { display: 'block', fontSize: 12, fontWeight: 600, color: '#555', marginBottom: 4 },
  input: (disabled) => ({
    width: '100%', padding: '8px 10px', borderRadius: 5, fontSize: 14,
    border: '1px solid #ccd6e0',
    color: disabled ? '#a0aec0' : '#1a202c',
    background: disabled ? '#f7f9fc' : '#fff',
    cursor: disabled ? 'not-allowed' : 'text',
  }),
  textarea: (disabled) => ({
    width: '100%', padding: '8px 10px', borderRadius: 5, fontSize: 14,
    border: '1px solid #ccd6e0',
    color: disabled ? '#a0aec0' : '#1a202c',
    background: disabled ? '#f7f9fc' : '#fff',
    cursor: disabled ? 'not-allowed' : 'text',
    resize: 'vertical', minHeight: 72, lineHeight: 1.5,
    fontFamily: 'inherit',
  }),
  hint: { fontSize: 11, color: '#888', marginTop: 4 },
  btnRow: { display: 'flex', gap: 10, marginTop: 20, alignItems: 'center' },
  btnSubmit: (disabled) => ({
    padding: '10px 28px',
    background: disabled ? '#7a9abf' : '#003087',
    color: '#fff', border: 'none', borderRadius: 5, fontSize: 14,
    fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
  }),
}

const DEFAULTS = {
  sector: 'Healthcare Services',
  deal_size_mm: 200,
  geography: 'Midwest',
  ownership: 'Private',
  profile_description: 'Mid-market, private, regional, strong EBITDA margins',
}

export default function TargetForm({ onRunStarted, loading, formLocked, historicalTarget }) {
  const [form, setForm] = useState(DEFAULTS)
  // editMode: false = locked (viewing history), true = editable (new or editing)
  const [editMode, setEditMode] = useState(true)

  // When a historical run is selected from the sidebar, populate + lock the form
  useEffect(() => {
    if (historicalTarget) {
      setForm({
        sector: historicalTarget.sector || DEFAULTS.sector,
        deal_size_mm: historicalTarget.deal_size_mm || DEFAULTS.deal_size_mm,
        geography: historicalTarget.geography || DEFAULTS.geography,
        ownership: historicalTarget.ownership || DEFAULTS.ownership,
        profile_description: historicalTarget.profile_description || DEFAULTS.profile_description,
      })
      setEditMode(false)
    } else {
      // historicalTarget cleared by sidebar "+ New" — always unlock
      setEditMode(true)
    }
  }, [historicalTarget])

  // Fields + submit are disabled while a run is in progress OR after it completes.
  // Only the sidebar "+ New" button resets formLocked and re-enables the form.
  const fieldDisabled = formLocked || !editMode || loading
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (formLocked || !editMode || loading) return  // guard against double-submit race
    const body = { ...form, deal_size_mm: parseFloat(form.deal_size_mm) }
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    onRunStarted(data.run_id, data.stream_url)
  }

  const isViewing = !editMode && !!historicalTarget

  return (
    <div style={S.card}>
      <div style={S.header}>
        <div style={S.title}>Target Company Profile</div>
        {isViewing && <span style={S.historicalBadge}>Viewing historical run</span>}
      </div>

      <form onSubmit={handleSubmit}>
        <div style={S.grid}>
          <div>
            <label style={S.label}>Sector</label>
            <input style={S.input(fieldDisabled)} value={form.sector} onChange={set('sector')} disabled={fieldDisabled} required />
          </div>
          <div>
            <label style={S.label}>Enterprise Value ($M)</label>
            <input style={S.input(fieldDisabled)} type="number" value={form.deal_size_mm} onChange={set('deal_size_mm')} disabled={fieldDisabled} required />
          </div>
          <div>
            <label style={S.label}>Geography</label>
            <input style={S.input(fieldDisabled)} value={form.geography} onChange={set('geography')} disabled={fieldDisabled} required />
          </div>
          <div>
            <label style={S.label}>Ownership</label>
            <input style={S.input(fieldDisabled)} value={form.ownership} onChange={set('ownership')} disabled={fieldDisabled} required />
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <label style={S.label}>Profile Description</label>
          <textarea
            style={S.textarea(fieldDisabled)}
            value={form.profile_description}
            onChange={set('profile_description')}
            placeholder="e.g. Mid-market, private, regional, strong EBITDA margins"
            disabled={fieldDisabled}
            required
          />
          {editMode && !loading && (
            <div style={S.hint}>
              Describe what makes this target attractive — margins, growth profile, competitive position, customer mix, etc.
            </div>
          )}
        </div>

        <div style={S.btnRow}>
          <button
            type="submit"
            style={S.btnSubmit(fieldDisabled)}
            disabled={fieldDisabled}
          >
            {loading ? 'Running Analysis…' : 'Identify Acquirers'}
          </button>
        </div>
      </form>
    </div>
  )
}
