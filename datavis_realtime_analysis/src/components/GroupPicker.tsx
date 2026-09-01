import type { Group } from '../lib/zarrData'
import { panel, control, label } from '../lib/styles'

type Props = {
  groups: Group[]
  active: string // active group path
  onChange: (path: string) => void
  onDelete: (id: string) => void
}

// Rail section: pick which embedding ("group") to view — the root store or any
// nested view store (e.g. a re-clustered cell subset). Hidden when there's nothing
// to switch between (a single group), so it stays out of the way by default.
export default function GroupPicker({ groups, active, onChange, onDelete }: Props) {
  if (groups.length < 2) return null
  const activeEntry = groups.find(g => g.path === active)
  return (
    <div style={panel}>
      <label style={label}>View</label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <select value={active} onChange={e => onChange(e.target.value)} style={control}>
          {groups.map(g => (
            <option key={g.id} value={g.path}>
              {g.label}
            </option>
          ))}
        </select>
        {activeEntry?.path && (
          <button
            onClick={() => onDelete(activeEntry.id)}
            style={deleteStyle}
            title="Delete this view from the store"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  )
}

const deleteStyle: React.CSSProperties = {
  background: 'none',
  color: '#a66',
  border: '1px solid #533',
  borderRadius: 4,
  padding: '2px 6px',
  fontSize: 11,
  cursor: 'pointer',
}
