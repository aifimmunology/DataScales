import { useState } from 'react'
import { panel, control, label, exprGradient } from '../lib/styles'

type Props = {
  genes: string[]
  active: string | null
  range: [number, number] | null
  error: string | null
  onChange: (gene: string | null) => void
}

const MAX_MATCHES = 50

export default function GenePicker({ genes, active, range, error, onChange }: Props) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)

  if (genes.length === 0) return null

  const q = query.trim().toLowerCase()
  const matches = q ? genes.filter(g => g.toLowerCase().includes(q)).slice(0, MAX_MATCHES) : []

  const pick = (gene: string) => {
    onChange(gene)
    setQuery('')
    setOpen(false)
  }

  return (
    <div style={panelStyle}>
      <label style={label}>Color by gene</label>
      {active ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, color: '#ffe94d' }}>{active}</span>
          <button onClick={() => onChange(null)} style={clearStyle}>✕</button>
        </div>
      ) : (
        <input
          value={query}
          placeholder={`Search ${genes.length.toLocaleString()} genes…`}
          onChange={e => {
            setQuery(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          style={inputStyle}
        />
      )}
      {open && matches.length > 0 && (
        <div style={listStyle}>
          {matches.map(g => (
            <div key={g} style={itemStyle} onClick={() => pick(g)}>
              {g}
            </div>
          ))}
        </div>
      )}
      {active && range && (
        <div style={{ marginTop: 6 }}>
          <div style={gradientStyle} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }}>
            <span>{fmt(range[0])}</span>
            <span>{fmt(range[1])}</span>
          </div>
        </div>
      )}
      {error && <div style={{ marginTop: 6, fontSize: 11, color: '#f88' }}>{error}</div>}
    </div>
  )
}

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2))

const panelStyle: React.CSSProperties = { ...panel, width: 180 }

const inputStyle: React.CSSProperties = { ...control, width: '100%', boxSizing: 'border-box' }

const listStyle: React.CSSProperties = {
  marginTop: 4,
  maxHeight: 180,
  overflowY: 'auto',
  border: '1px solid #333',
  borderRadius: 4,
  background: '#1c1c1c',
}

const itemStyle: React.CSSProperties = {
  padding: '4px 8px',
  fontSize: 13,
  cursor: 'pointer',
}

const clearStyle: React.CSSProperties = {
  background: 'none',
  color: '#888',
  border: 'none',
  cursor: 'pointer',
  fontSize: 13,
  padding: 0,
}

const gradientStyle: React.CSSProperties = {
  height: 8,
  borderRadius: 4,
  background: exprGradient,
}
