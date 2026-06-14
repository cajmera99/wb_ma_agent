/**
 * Renders one acquirer rationale card in the results panel.
 * Collapsible — shows header by default, expands to full rationale.
 */
import { useState } from 'react'

const CONVICTION_COLORS = { High: '#1E8449', Medium: '#D68910', Low: '#C0392B' }

const S = {
  card: {
    background: '#fff',
    border: '1px solid #e2e8f0',
    borderRadius: 8,
    marginBottom: 10,
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    padding: '12px 16px',
    cursor: 'pointer',
    userSelect: 'none',
  },
  rank: {
    width: 28, height: 28, borderRadius: '50%', background: '#003087',
    color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 12, fontWeight: 700, flexShrink: 0, marginRight: 12,
  },
  name: { fontSize: 14, fontWeight: 700, color: '#1a202c', flex: 1 },
  type: { fontSize: 11, color: '#718096', marginLeft: 8 },
  score: { fontSize: 13, fontWeight: 600, color: '#003087', marginLeft: 16 },
  badge: (conviction) => ({
    marginLeft: 12, padding: '2px 8px', borderRadius: 10, fontSize: 11,
    fontWeight: 600, background: CONVICTION_COLORS[conviction] || '#718096',
    color: '#fff',
  }),
  toggle: { marginLeft: 10, color: '#718096', fontSize: 12 },
  body: { padding: '0 16px 16px' },
  section: { marginTop: 14 },
  sLabel: { fontSize: 11, fontWeight: 700, color: '#003087', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 5 },
  text: { fontSize: 13, color: '#2d3748', lineHeight: 1.6 },
  riskRow: { display: 'flex', gap: 8, marginBottom: 5, alignItems: 'flex-start' },
  severity: (s) => ({
    fontSize: 10, fontWeight: 700, padding: '1px 6px', borderRadius: 3,
    background: { High: '#FDECEA', Medium: '#FEF9E7', Low: '#EAFAF1' }[s] || '#eee',
    color: { High: '#C0392B', Medium: '#D68910', Low: '#1E8449' }[s] || '#333',
    whiteSpace: 'nowrap',
  }),
  divider: { borderTop: '1px solid #edf2f7', marginTop: 14 },
  dealTable: { width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 6 },
  th: { textAlign: 'left', padding: '4px 8px', background: '#EAF0F8', color: '#003087', fontWeight: 600 },
  td: { padding: '4px 8px', borderBottom: '1px solid #f0f4f8' },
}

export default function AcquirerCard({ rationale }) {
  const [open, setOpen] = useState(false)
  const { acquirer_name, acquirer_type, composite_score, conviction_level, rank } = rationale

  if (rationale.error) {
    return (
      <div style={{ ...S.card, borderLeft: '3px solid #f85149' }}>
        <div style={S.header}>
          <div style={S.rank}>{rank}</div>
          <div style={S.name}>{acquirer_name}</div>
          <span style={{ fontSize: 12, color: '#f85149' }}>Generation failed</span>
        </div>
      </div>
    )
  }

  return (
    <div style={{ ...S.card, borderLeft: `3px solid ${CONVICTION_COLORS[conviction_level] || '#003087'}` }}>
      <div style={S.header} onClick={() => setOpen((o) => !o)}>
        <div style={S.rank}>{rank}</div>
        <div style={S.name}>
          {acquirer_name}
          <span style={S.type}>{acquirer_type}</span>
        </div>
        <span style={S.score}>{composite_score?.toFixed(1)}/100</span>
        <span style={S.badge(conviction_level)}>{conviction_level}</span>
        <span style={S.toggle}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div style={S.body}>
          <div style={S.divider} />

          <div style={S.section}>
            <div style={S.sLabel}>Acquirer Overview</div>
            <div style={S.text}>{rationale.acquirer_overview}</div>
          </div>

          <div style={S.section}>
            <div style={S.sLabel}>Strategic Fit Thesis</div>
            <div style={S.text}>{rationale.strategic_fit_thesis}</div>
          </div>

          {rationale.precedent_deals?.length > 0 && (
            <div style={S.section}>
              <div style={S.sLabel}>Precedent Activity</div>
              <table style={S.dealTable}>
                <thead>
                  <tr>
                    {['Target', 'Year', 'Size ($M)', 'Type', 'EV/EBITDA', 'Outcome'].map((h) => (
                      <th key={h} style={S.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rationale.precedent_deals.map((d, i) => (
                    <tr key={i}>
                      <td style={S.td}>{d.target_company}</td>
                      <td style={S.td}>{d.deal_year}</td>
                      <td style={S.td}>${d.deal_size_mm?.toFixed(0)}</td>
                      <td style={S.td}>{d.deal_type}</td>
                      <td style={S.td}>{d.ev_ebitda_multiple ? `${d.ev_ebitda_multiple}x` : 'N/A'}</td>
                      <td style={S.td}>{d.outcome}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {rationale.valuation_context && (
            <div style={S.section}>
              <div style={S.sLabel}>Valuation Context</div>
              <div style={S.text}>
                Market EV/EBITDA: <b>{rationale.valuation_context.median_ev_ebitda ?? 'N/A'}x</b> &nbsp;|&nbsp;
                EV/Revenue: <b>{rationale.valuation_context.median_ev_revenue ?? 'N/A'}x</b> &nbsp;|&nbsp;
                Based on <b>{rationale.valuation_context.deal_count_in_range}</b> transactions.
                {rationale.valuation_context.note && ` ${rationale.valuation_context.note}`}
              </div>
            </div>
          )}

          {rationale.risk_flags?.length > 0 && (
            <div style={S.section}>
              <div style={S.sLabel}>Risk Flags</div>
              {rationale.risk_flags.map((r, i) => (
                <div key={i} style={S.riskRow}>
                  <span style={S.severity(r.severity)}>{r.severity}</span>
                  <span style={S.text}><b>{r.risk_type}</b> — {r.description}</span>
                </div>
              ))}
            </div>
          )}

          <div style={S.section}>
            <div style={S.sLabel}>Conviction</div>
            <div style={{ ...S.text, color: CONVICTION_COLORS[conviction_level] }}>
              <b>{conviction_level}</b> — {rationale.conviction_rationale}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
