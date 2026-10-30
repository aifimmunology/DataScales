import { useEffect, useState, type PointerEvent, type ReactNode } from 'react'
import { Icon } from '@iconify/react'
import { DIVIDER, HEADER_HEIGHT } from '../lib/styles'
import { useDragWidth } from '../hooks/useDragWidth'

export type RailSection = {
  id: string
  icon: string
  title: string
  badge?: boolean
  badgeColor?: string
  content: ReactNode
  aside?: { icon: string; title: string; content: ReactNode }
}

const DRAWER_WIDTH = 320
const MIN_WIDTH = 240
const MAX_WIDTH = 720

/** Icon rail with one open section at a time, plus an optional second drawer beside it.
 * Drawers overlay the canvas and are resized by dragging their right edge. */
export default function SideRail({ sections }: { sections: RailSection[] }) {
  const [active, setActive] = useState<string | null>(null)
  const [asideOpen, setAsideOpen] = useState(false)
  const [mainWidth, dragMain] = useDragWidth(DRAWER_WIDTH, MIN_WIDTH, MAX_WIDTH)
  const [asideWidth, dragAside] = useDragWidth(DRAWER_WIDTH, MIN_WIDTH, MAX_WIDTH)
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
            style={{ ...railBtnStyle, ...(active === s.id ? activeStyle : {}) }}
          >
            <Icon icon={s.icon} width={26} height={26} />
            {s.badge && <span style={{ ...badgeStyle, background: s.badgeColor ?? '#ffe94d' }} />}
          </button>
        ))}
      </div>
      {current && (
        <Drawer title={current.title} left={HEADER_HEIGHT} width={mainWidth} onResize={dragMain} onClose={() => setActive(null)}>
          {current.content}
          {current.aside && (
            <button
              title={current.aside.title}
              onClick={() => setAsideOpen(o => !o)}
              style={{ ...asideTabStyle, ...(asideOpen ? activeStyle : {}) }}
            >
              <Icon icon={current.aside.icon} width={20} height={20} />
            </button>
          )}
        </Drawer>
      )}
      {current?.aside && asideOpen && (
        <Drawer
          title={current.aside.title}
          left={HEADER_HEIGHT + mainWidth}
          width={asideWidth}
          onResize={dragAside}
          onClose={() => setAsideOpen(false)}
        >
          {current.aside.content}
        </Drawer>
      )}
    </>
  )
}

type DrawerProps = {
  title: string
  left: number
  width: number
  onResize: (e: PointerEvent<HTMLElement>) => void
  onClose: () => void
  children: ReactNode
}

function Drawer({ title, left, width, onResize, onClose, children }: DrawerProps) {
  return (
    <div style={{ ...drawerStyle, left, width }}>
      <div style={drawerHeaderStyle}>
        <span style={{ fontSize: 14, fontWeight: 600 }}>{title}</span>
        <button onClick={onClose} title="Close panel" style={closeBtnStyle}>
          <Icon icon="material-symbols-light:left-panel-close" width={22} height={22} />
        </button>
      </div>
      <div style={drawerBodyStyle}>{children}</div>
      <div style={handleStyle} onPointerDown={onResize} title="Drag to resize" />
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

const activeStyle: React.CSSProperties = { color: '#fff', background: 'rgba(255, 255, 255, 0.08)' }

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

const badgeStyle: React.CSSProperties = { position: 'absolute', top: 12, right: 12, width: 8, height: 8, borderRadius: 4 }

const drawerStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  bottom: 0,
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

const drawerBodyStyle: React.CSSProperties = { overflowY: 'auto', padding: 10, display: 'flex', flexDirection: 'column', gap: 10 }

const handleStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  bottom: 0,
  right: 0,
  width: 6,
  cursor: 'col-resize',
  touchAction: 'none',
  zIndex: 1,
}

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#aaa',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
}
