import { DIVIDER } from '../lib/styles'

export const HEADER_HEIGHT = 55 // also the side rail width, matching the spatial explorer

// Always-black top bar in the spatial-data-explorer style: logo + slash-path
// title with the brand-color pulsing segment.
export default function Header() {
  return (
    <div style={barStyle}>
      <div style={{ display: 'flex', alignItems: 'center', minWidth: 0 }}>
        <img src="/ai-new-logo.png" alt="Allen Institute logo" height={42} style={{ marginRight: 12, flexShrink: 0 }} />
        <span style={titleStyle}>
          allen institute/<span className="allen-pulse">immunology/</span>datascale umap
        </span>
      </div>
    </div>
  )
}

const barStyle: React.CSSProperties = {
  height: HEADER_HEIGHT,
  flexShrink: 0,
  background: '#000',
  borderBottom: `1px solid ${DIVIDER}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 13px',
}

const titleStyle: React.CSSProperties = {
  color: '#fff',
  fontSize: '1.25rem',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}
