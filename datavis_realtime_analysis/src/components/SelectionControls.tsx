import { useState } from 'react'
import { panel, control, primaryBtn } from '../lib/styles'

type Props = {
  selecting: boolean
  onToggle: () => void
  count: number
  onSubmit: (name: string) => void
  onClear: () => void
}

// Rail section: toggle lasso mode, then name + submit the selection to the GPU, or clear it.
export default function SelectionControls({ selecting, onToggle, count, onSubmit, onClear }: Props) {
  const [name, setName] = useState('')
  return (
    <div style={panelStyle}>
      <button onClick={onToggle} style={{ ...btnStyle, ...(selecting ? activeStyle : {}) }}>
        {selecting ? '✏️ Drawing — drag to lasso' : '◌ Select cells (lasso)'}
      </button>

      {count > 0 && (
        <>
          <div style={{ fontSize: 12, color: '#ddd' }}>{count.toLocaleString()} cells selected</div>
          <input
            value={name}
            placeholder="new view name…"
            onChange={e => setName(e.target.value)}
            style={{ ...btnStyle, cursor: 'text' }}
          />
          <button
            onClick={() => {
              onSubmit(name.trim())
              setName('')
            }}
            style={primaryBtn}
          >
            Generate New UMAP
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
