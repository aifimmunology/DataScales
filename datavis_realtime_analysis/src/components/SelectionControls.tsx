import { useState } from 'react'
import { button, column, dangerButton, primaryBtn } from '../lib/styles'

type Props = {
  selecting: boolean
  onToggle: () => void
  count: number
  onSubmit: (name: string) => void
  onClear: () => void
}

export default function SelectionControls({ selecting, onToggle, count, onSubmit, onClear }: Props) {
  const [name, setName] = useState('')
  return (
    <div style={column(8)}>
      <button onClick={onToggle} style={{ ...button, ...(selecting ? activeStyle : {}) }}>
        {selecting ? '✏️ Drawing — drag to lasso' : '◌ Select cells (lasso)'}
      </button>
      {count > 0 && (
        <>
          <div style={{ fontSize: 12, color: '#ddd' }}>{count.toLocaleString()} cells selected</div>
          <input value={name} placeholder="new view name…" onChange={e => setName(e.target.value)} style={{ ...button, cursor: 'text' }} />
          <button
            onClick={() => {
              onSubmit(name.trim())
              setName('')
            }}
            style={primaryBtn}
          >
            Generate New UMAP
          </button>
          <button onClick={onClear} style={dangerButton}>
            Clear
          </button>
        </>
      )}
    </div>
  )
}

const activeStyle: React.CSSProperties = { background: '#3a3411', borderColor: '#c9b81f', color: '#ffe94d' }
