import { LEVELS, type Level, colorForCode } from '../lib/zarrData'
import { panel, control, label, rgb } from '../lib/styles'

type Props = {
  level: Level
  onLevelChange: (level: Level) => void
  categories: string[] | null
}

// Top-right overlay: pick which AIFI level colors the UMAP + show the key.
export default function Legend({ level, onLevelChange, categories }: Props) {
  return (
    <div style={panelStyle}>
      <label style={label}>Color by</label>
      <select
        value={level}
        onChange={e => onLevelChange(e.target.value as Level)}
        style={control}
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
  ...panel,
  position: 'absolute',
  top: 12,
  right: 12,
  zIndex: 20, // above the lasso svg overlay
  maxHeight: 'calc(100vh - 24px)',
  overflowY: 'auto',
}

const swatchStyle: React.CSSProperties = {
  width: 12,
  height: 12,
  borderRadius: 2,
  flexShrink: 0,
}
