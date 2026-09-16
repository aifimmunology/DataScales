import { type Level, colorForCode } from '../lib/zarrData'
import { panel, control, label, rgb } from '../lib/styles'

type Props = {
  levels: string[] // label sets discovered in the store (per group)
  level: Level
  onLevelChange: (level: Level) => void
  categories: string[] | null
  selected: Set<number> // category codes currently cluster-selected
  onCategoryClick: (code: number) => void
  error: string | null
  onRetry: () => void
}

// Top-right overlay: pick which label set colors the UMAP + show the key.
// Clicking a key row toggles selecting every cell of that category (cluster select).
export default function Legend({ levels, level, onLevelChange, categories, selected, onCategoryClick, error, onRetry }: Props) {
  if (levels.length === 0) return null
  return (
    <div style={panelStyle}>
      <label style={label}>Color by</label>
      <select
        value={level}
        onChange={e => onLevelChange(e.target.value as Level)}
        style={control}
      >
        {levels.map(l => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>

      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {error ? (
          <span
            style={{ color: '#f88', fontSize: 12, cursor: 'pointer' }}
            onClick={onRetry}
            title={error}
          >
            failed to load — retry
          </span>
        ) : categories === null ? (
          <span style={{ color: '#888', fontSize: 12 }}>Loading…</span>
        ) : (
          categories.map((name, code) => (
            <div
              key={name}
              onClick={() => onCategoryClick(code)}
              title="click to select these cells"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                cursor: 'pointer',
                padding: '1px 4px',
                borderRadius: 3,
                background: selected.has(code) ? '#3a3411' : 'transparent',
              }}
            >
              <span style={{ ...swatchStyle, background: rgb(colorForCode(code)) }} />
              <span style={{ fontSize: 12, color: selected.has(code) ? '#ffe94d' : '#ddd' }}>{name}</span>
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
