import { LEVELS, type Level, colorForCode, type RGB } from '../lib/zarrData'

type Props = {
  level: Level
  onLevelChange: (level: Level) => void
  categories: string[] | null
}

const rgb = (c: RGB) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`

// Top-right overlay: pick which AIFI level colors the UMAP + show the key.
export default function Legend({ level, onLevelChange, categories }: Props) {
  return (
    <div style={panelStyle}>
      <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
        Color by
      </label>
      <select
        value={level}
        onChange={e => onLevelChange(e.target.value as Level)}
        style={selectStyle}
      >
        {LEVELS.map(l => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>

      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {categories === null ? (
          <span style={{ color: '#888', fontSize: 12 }}>Loading…</span>
        ) : (
          categories.map((name, code) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ ...swatchStyle, background: rgb(colorForCode(code)) }} />
              <span style={{ fontSize: 12, color: '#ddd' }}>{name}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  right: 12,
  zIndex: 20, // above the lasso svg overlay
  background: 'rgba(20, 20, 20, 0.85)',
  border: '1px solid #333',
  borderRadius: 6,
  padding: '10px 12px',
  color: '#ddd',
  fontFamily: 'system-ui, sans-serif',
  maxHeight: 'calc(100vh - 24px)',
  overflowY: 'auto',
}

const selectStyle: React.CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '4px 6px',
  fontSize: 13,
}

const swatchStyle: React.CSSProperties = {
  width: 12,
  height: 12,
  borderRadius: 2,
  flexShrink: 0,
}
