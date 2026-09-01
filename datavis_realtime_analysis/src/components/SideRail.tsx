import { useState, type ReactNode } from 'react'
import { Icon } from '@iconify/react'
import { DIVIDER } from '../lib/styles'
import { HEADER_HEIGHT } from './Header'

export type RailSection = {
  id: string
  icon: string
  title: string
  badge?: boolean // e.g. a GPU run is active
  content: ReactNode
}

type Props = { sections: RailSection[] }

// Spatial-explorer-style icon rail: one section open at a time; the drawer
// overlays the canvas (no deck resize) so the plot and legend stay put.
export default function SideRail({ sections }: Props) {
  const [active, setActive] = useState<string | null>(null)
  const current = sections.find(s => s.id === active)
  return (
    <>
      <div style={railStyle}>
        {sections.map(s => (
          <button
            key={s.id}
            title={s.title}
            onClick={() => setActive(a => (a === s.id ? null : s.id))}
            style={{ ...railBtnStyle, ...(active === s.id ? railBtnActiveStyle : {}) }}
          >
            <Icon icon={s.icon} width={26} height={26} />
            {s.badge && <span style={badgeStyle} />}
          </button>
        ))}
      </div>
      {current && (
        <div style={drawerStyle}>
          <div style={drawerHeaderStyle}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{current.title}</span>
            <button onClick={() => setActive(null)} title="Close panel" style={closeBtnStyle}>
              <Icon icon="material-symbols-light:left-panel-close" width={22} height={22} />
            </button>
          </div>
          <div style={{ overflowY: 'auto', padding: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
            {current.content}
          </div>
        </div>
      )}
    </>
  )
}

const railStyle: React.CSSProperties = {
  width: HEADER_HEIGHT,
  flexShrink: 0,
  background: '#000',
  borderRight: `1px solid ${DIVIDER}`,
  display: 'flex',
  flexDirection: 'column',
}

const railBtnStyle: React.CSSProperties = {
  position: 'relative',
  width: HEADER_HEIGHT,
  height: HEADER_HEIGHT,
  background: 'none',
  border: 'none',
  color: '#aaa',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

const railBtnActiveStyle: React.CSSProperties = {
  color: '#fff',
  background: 'rgba(255, 255, 255, 0.08)',
}

const badgeStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  right: 12,
  width: 8,
  height: 8,
  borderRadius: 4,
  background: '#ffe94d',
}

// Overlays the canvas, anchored beside the rail.
const drawerStyle: React.CSSProperties = {
  position: 'absolute',
  left: HEADER_HEIGHT,
  top: 0,
  bottom: 0,
  width: 320,
  zIndex: 25,
  background: 'rgba(18, 18, 18, 0.97)',
  borderRight: `1px solid ${DIVIDER}`,
  display: 'flex',
  flexDirection: 'column',
  color: '#ddd',
}

const drawerHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '8px 8px 8px 12px',
  borderBottom: `1px solid ${DIVIDER}`,
}

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#aaa',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
}
