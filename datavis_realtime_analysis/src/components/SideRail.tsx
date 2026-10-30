import { useEffect, useState, type ReactNode } from 'react'
import { Icon } from '@iconify/react'
import { DIVIDER, HEADER_HEIGHT } from '../lib/styles'

export type RailAside = { icon: string; title: string; content: ReactNode }

export type RailSection = {
  id: string
  icon: string
  title: string
  badge?: boolean // e.g. a GPU run is active
  badgeColor?: string // overrides the default yellow, e.g. red for a GPU access problem
  content: ReactNode
  aside?: RailAside // optional second drawer, opened from a small tab under the content
}

type Props = { sections: RailSection[] }

const DRAWER_WIDTH = 320

// Spatial-explorer-style icon rail: one section open at a time; drawers overlay
// the canvas (no deck resize) so the plot and legend stay put.
export default function SideRail({ sections }: Props) {
  const [active, setActive] = useState<string | null>(null)
  const [asideOpen, setAsideOpen] = useState(false)
  useEffect(() => setAsideOpen(false), [active])
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
            {s.badge && (
              <span style={s.badgeColor ? { ...badgeStyle, background: s.badgeColor } : badgeStyle} />
            )}
          </button>
        ))}
      </div>
      {current && (
        <Drawer title={current.title} left={HEADER_HEIGHT} onClose={() => setActive(null)}>
          {current.content}
          {current.aside && (
            <button
              title={current.aside.title}
              onClick={() => setAsideOpen(o => !o)}
              style={{ ...asideTabStyle, ...(asideOpen ? railBtnActiveStyle : {}) }}
            >
              <Icon icon={current.aside.icon} width={20} height={20} />
            </button>
          )}
        </Drawer>
      )}
      {current?.aside && asideOpen && (
        <Drawer title={current.aside.title} left={HEADER_HEIGHT + DRAWER_WIDTH} onClose={() => setAsideOpen(false)}>
          {current.aside.content}
        </Drawer>
      )}
    </>
  )
}

function Drawer({ title, left, onClose, children }: { title: string; left: number; onClose: () => void; children: ReactNode }) {
  return (
    <div style={{ ...drawerStyle, left }}>
      <div style={drawerHeaderStyle}>
        <span style={{ fontSize: 14, fontWeight: 600 }}>{title}</span>
        <button onClick={onClose} title="Close panel" style={closeBtnStyle}>
          <Icon icon="material-symbols-light:left-panel-close" width={22} height={22} />
        </button>
      </div>
      <div style={drawerBodyStyle}>{children}</div>
    </div>
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

const asideTabStyle: React.CSSProperties = {
  alignSelf: 'flex-start',
  background: 'none',
  border: `1px solid ${DIVIDER}`,
  borderRadius: 6,
  color: '#aaa',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  padding: 4,
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

const drawerStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  bottom: 0,
  width: DRAWER_WIDTH,
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

const drawerBodyStyle: React.CSSProperties = {
  overflowY: 'auto',
  padding: 10,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
}

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#aaa',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
}
