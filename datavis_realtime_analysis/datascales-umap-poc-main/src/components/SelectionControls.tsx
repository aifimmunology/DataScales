type Props = {
  selecting: boolean
  onToggle: () => void
  count: number
  onDownload: () => void
  onClear: () => void
}

// Top-left overlay: toggle lasso mode, then download / clear the selection.
export default function SelectionControls({
  selecting,
  onToggle,
  count,
  onDownload,
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
          <button onClick={onClear} style={{ ...btnStyle, color: '#f88', borderColor: '#622' }}>
            Clear
          </button>
        </>
      )}
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  left: 12,
  zIndex: 20, // above the lasso svg overlay so the buttons stay clickable
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  background: 'rgba(20, 20, 20, 0.85)',
  border: '1px solid #333',
  borderRadius: 6,
  padding: '10px 12px',
  fontFamily: 'system-ui, sans-serif',
}

const btnStyle: React.CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '6px 10px',
  fontSize: 13,
  cursor: 'pointer',
  textAlign: 'left',
}

const activeStyle: React.CSSProperties = {
  background: '#3a3411',
  borderColor: '#c9b81f',
  color: '#ffe94d',
}
