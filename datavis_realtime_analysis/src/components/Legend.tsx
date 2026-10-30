import { type RGB, colorForCode, UNLABELED_CODE, UNLABELED_SWATCH } from '../lib/palette'
import { panel, label, rgb } from '../lib/styles'
import ComboBox from './ComboBox'

type Props = {
  levels: string[]
  labelFor: (level: string) => string
  level: string
  onLevelChange: (level: string) => void
  categories: string[] | null
  unlabeled: boolean
  selected: Set<number>
  onCategoryClick: (code: number) => void
  error: string | null
  onRetry: () => void
}

export default function Legend({ levels, labelFor, level, onLevelChange, categories, unlabeled, selected, onCategoryClick, error, onRetry }: Props) {
  if (levels.length === 0) return null
  return (
    <div style={panelStyle}>
      <label style={label}>Color by</label>
      <ComboBox
        options={levels.map(l => ({ value: l, label: labelFor(l) }))}
        value={level}
        placeholder="search columns…"
        onPick={onLevelChange}
      />

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
          <>
            {categories.map((name, code) => (
              <Row key={name} name={name} color={colorForCode(code)} active={selected.has(code)} onClick={() => onCategoryClick(code)} />
            ))}
            {unlabeled && (
              <Row
                name="unlabeled"
                color={UNLABELED_SWATCH}
                active={selected.has(UNLABELED_CODE)}
                onClick={() => onCategoryClick(UNLABELED_CODE)}
                title="click to select every cell without a label"
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}

type RowProps = { name: string; color: RGB; active: boolean; onClick: () => void; title?: string }

function Row({ name, color, active, onClick, title = 'click to select these cells' }: RowProps) {
  return (
    <div onClick={onClick} title={title} style={{ ...rowStyle, background: active ? '#3a3411' : 'transparent' }}>
      <span style={{ ...swatchStyle, background: rgb(color) }} />
      <span style={{ fontSize: 12, color: active ? '#ffe94d' : '#ddd' }}>{name}</span>
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  ...panel,
  position: 'absolute',
  top: 12,
  right: 12,
  zIndex: 20,
  maxHeight: 'calc(100vh - 24px)',
  overflowY: 'auto',
  minWidth: 180,
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  cursor: 'pointer',
  padding: '1px 4px',
  borderRadius: 3,
}

const swatchStyle: React.CSSProperties = {
  width: 12,
  height: 12,
  borderRadius: 2,
  flexShrink: 0,
}
