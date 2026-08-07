import type { Group } from '../lib/zarrData'

type Props = {
  groups: Group[]
  active: string // active group path
  onChange: (path: string) => void
}

// Top-left overlay: pick which embedding ("group") to view — the root store or any
// nested view store (e.g. a re-clustered cell subset). Hidden when there's nothing
// to switch between (a single group), so it stays out of the way by default.
export default function GroupPicker({ groups, active, onChange }: Props) {
  if (groups.length < 2) return null
  return (
    <div style={panelStyle}>
      <label style={{ display: 'block', fontSize: 11, color: '#888', marginBottom: 4 }}>
        View
      </label>
      <select value={active} onChange={e => onChange(e.target.value)} style={selectStyle}>
        {groups.map(g => (
          <option key={g.id} value={g.path}>
            {g.label}
          </option>
        ))}
      </select>
    </div>
  )
}

// Positioned by the top-left stack wrapper in Umap.tsx, beneath the selection tool.
const panelStyle: React.CSSProperties = {
  background: 'rgba(20, 20, 20, 0.85)',
  border: '1px solid #333',
  borderRadius: 6,
  padding: '10px 12px',
  color: '#ddd',
  fontFamily: 'system-ui, sans-serif',
}

const selectStyle: React.CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '4px 6px',
  fontSize: 13,
}
