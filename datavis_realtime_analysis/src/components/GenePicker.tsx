import { panel, label, exprGradient } from '../lib/styles'
import ComboBox from './ComboBox'

type Props = {
  genes: string[]
  active: string | null
  range: [number, number] | null
  error: string | null
  warning: string | null
  onChange: (gene: string | null) => void
}

const MAX_MATCHES = 50

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2))

export default function GenePicker({ genes, active, range, error, warning, onChange }: Props) {
  return (
    <div style={panel}>
      <label style={label}>Color by gene</label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ flex: 1 }}>
          <ComboBox
            options={genes.map(g => ({ value: g }))}
            value={active ?? ''}
            limit={MAX_MATCHES}
            placeholder={`Search ${genes.length.toLocaleString()} genes…`}
            onPick={onChange}
          />
        </div>
        {active && (
          <button onClick={() => onChange(null)} style={clearStyle} title="Clear gene">
            ✕
          </button>
        )}
      </div>
      {active && range && (
        <div style={{ marginTop: 6 }}>
          <div style={gradientStyle} />
          <div style={rangeStyle}>
            <span>{fmt(range[0])}</span>
            <span>{fmt(range[1])}</span>
          </div>
        </div>
      )}
      {error && <div style={{ marginTop: 6, fontSize: 11, color: '#f88' }}>{error}</div>}
      {active && warning && <div style={{ marginTop: 6, fontSize: 11, color: '#cc5' }}>{warning}</div>}
    </div>
  )
}

const clearStyle: React.CSSProperties = { background: 'none', color: '#888', border: 'none', cursor: 'pointer', fontSize: 13, padding: 0 }
const gradientStyle: React.CSSProperties = { height: 8, borderRadius: 4, background: exprGradient }
const rangeStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888' }
