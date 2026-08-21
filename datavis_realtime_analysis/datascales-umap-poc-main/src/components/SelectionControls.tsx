import { panel, control } from '../lib/styles'

type Props = {
  selecting: boolean
  onToggle: () => void
  count: number
  onDownload: () => void
  onSubmit: () => void
  onClear: () => void
}

// Top-left overlay: toggle lasso mode, then download / submit / clear the selection.
export default function SelectionControls({
  selecting,
  onToggle,
  count,
  onDownload,
  onSubmit,
  onClear,
}: Props) {
  return (
    <div style={panelStyle}>
      <button onClick={onToggle} style={{ ...btnStyle, ...(selecting ? activeStyle : {}) }}>
        {selecting ? '✏️ Drawing — drag to lasso' : '◌ Select cells (lasso)'}
      </button>

      {count > 0 && (
        <>
          <div style={{ fontSize: 12, color: '#ddd' }}>
            {count.toLocaleString()} cells selected
          </div>
          <button onClick={onDownload} style={btnStyle}>
            ⬇ Download selection.json
          </button>
          <button onClick={onSubmit} style={btnStyle}>
            🚀 Submit to GPU
          </button>
          <button onClick={onClear} style={{ ...btnStyle, color: '#f88', borderColor: '#622' }}>
            Clear
          </button>
        </>
      )}
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  ...panel,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
}

const btnStyle: React.CSSProperties = {
  ...control,
  padding: '6px 10px',
  cursor: 'pointer',
  textAlign: 'left',
}

const activeStyle: React.CSSProperties = {
  background: '#3a3411',
  borderColor: '#c9b81f',
  color: '#ffe94d',
}
