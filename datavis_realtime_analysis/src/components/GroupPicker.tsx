import type { Group } from '../lib/api'
import { describeParams } from '../lib/runSettings'
import { panel, control, label, muted } from '../lib/styles'

type Props = {
  groups: Group[]
  active: string
  onChange: (path: string) => void
  onDelete: (id: string) => void
}

export default function GroupPicker({ groups, active, onChange, onDelete }: Props) {
  if (groups.length < 2) {
    return <span style={muted}>No saved views yet — lasso a selection and run it on the GPU.</span>
  }
  const current = groups.find(g => g.path === active)
  const params = describeParams(current?.params)
  return (
    <div style={panel}>
      <label style={label}>View</label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <select value={active} onChange={e => onChange(e.target.value)} style={{ ...control, flex: 1, minWidth: 0 }}>
          {groups.map(g => (
            <option key={g.id} value={g.path}>
              {g.label}
            </option>
          ))}
        </select>
        {current?.path && (
          <button onClick={() => onDelete(current.id)} style={deleteStyle} title="Delete this view from the store">
            ✕
          </button>
        )}
      </div>
      {params && <div style={paramsStyle}>{params}</div>}
    </div>
  )
}

const paramsStyle: React.CSSProperties = { fontSize: 11, color: '#888', marginTop: 6 }

const deleteStyle: React.CSSProperties = {
  background: 'none',
  color: '#a66',
  border: '1px solid #533',
  borderRadius: 4,
  padding: '2px 6px',
  fontSize: 11,
  cursor: 'pointer',
}
