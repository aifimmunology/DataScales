import { useRef, useState, type PointerEvent } from 'react'

type Pt = [number, number]

/** Captures a freehand polygon over the canvas; reports it in screen pixels. */
export default function LassoOverlay({ onPolygon }: { onPolygon: (screen: Pt[]) => void }) {
  const path = useRef<Pt[] | null>(null)
  const [drawn, setDrawn] = useState<Pt[]>([])

  const pos = (e: PointerEvent<HTMLDivElement>): Pt => {
    const r = e.currentTarget.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }
  const down = (e: PointerEvent<HTMLDivElement>) => {
    path.current = [pos(e)]
    setDrawn(path.current.slice())
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const move = (e: PointerEvent<HTMLDivElement>) => {
    if (!path.current) return
    path.current.push(pos(e))
    setDrawn(path.current.slice())
  }
  const up = () => {
    const p = path.current
    path.current = null
    setDrawn([])
    if (p && p.length >= 3) onPolygon(p)
  }

  return (
    <div style={overlayStyle} onPointerDown={down} onPointerMove={move} onPointerUp={up}>
      <svg style={svgStyle}>
        {drawn.length > 1 && (
          <polygon
            points={drawn.map(p => `${p[0]},${p[1]}`).join(' ')}
            fill="rgba(255, 240, 30, 0.12)"
            stroke="rgba(255, 240, 30, 0.9)"
            strokeWidth={1.5}
          />
        )}
      </svg>
    </div>
  )
}

const overlayStyle: React.CSSProperties = { position: 'absolute', inset: 0, zIndex: 10, cursor: 'crosshair' }
const svgStyle: React.CSSProperties = { position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }
