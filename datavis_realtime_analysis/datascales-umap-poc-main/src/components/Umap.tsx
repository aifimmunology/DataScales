import { useEffect, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import Legend from './Legend'
import SelectionControls from './SelectionControls'
import {
  loadUmapCoords,
  loadCategorical,
  loadBarcodes,
  colorForCode,
  STORE_ID,
  type Point,
  type Categorical,
  type Level,
} from '../lib/zarrData'
import { selectIndices, downloadSelection, type SelectionArtifact } from '../lib/selection'

type Sel = { mask: Uint8Array; indices: number[]; world: [number, number][] }

export default function Umap() {
  const [points, setPoints] = useState<Point[]>([])
  const [barcodes, setBarcodes] = useState<string[]>([])
  const [level, setLevel] = useState<Level>('AIFI_L1') // default color-by
  const [cat, setCat] = useState<Categorical | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // selection state
  const [selecting, setSelecting] = useState(false)
  const [selection, setSelection] = useState<Sel | null>(null)
  const [selVersion, setSelVersion] = useState(0) // bumps deck's getFillColor updateTrigger
  const [lassoScreen, setLassoScreen] = useState<[number, number][]>([])

  const deckRef = useRef<any>(null)
  const draggingRef = useRef(false)
  const pathRef = useRef<[number, number][]>([])

  // Coordinates + barcodes load once.
  useEffect(() => {
    Promise.all([loadUmapCoords(), loadBarcodes()])
      .then(([pts, bcs]) => {
        setPoints(pts)
        setBarcodes(bcs)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load UMAP data:', err)
        setError(String(err))
        setLoading(false)
      })
  }, [])

  // Category codes reload whenever the AIFI level changes.
  useEffect(() => {
    setCat(null)
    loadCategorical(level)
      .then(setCat)
      .catch(err => console.error(`Failed to load ${level}:`, err))
  }, [level])

  // ---- lasso drawing (screen space) -> selection (world space) ----
  const relPos = (e: React.PointerEvent<SVGSVGElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    draggingRef.current = true
    pathRef.current = [relPos(e)]
    setLassoScreen(pathRef.current.slice())
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!draggingRef.current) return
    pathRef.current.push(relPos(e))
    setLassoScreen(pathRef.current.slice())
  }

  const onPointerUp = () => {
    if (!draggingRef.current) return
    draggingRef.current = false
    const screen = pathRef.current
    pathRef.current = []
    setLassoScreen([])
    if (screen.length < 3) return

    // Unproject the drawn polygon to UMAP coords via the live viewport
    // (respects current pan/zoom), then point-in-polygon over the cells.
    const vp = deckRef.current?.deck?.getViewports?.()[0]
    if (!vp) return
    const world = screen.map(([x, y]) => {
      const u = vp.unproject([x, y])
      return [u[0], u[1]] as [number, number]
    })
    const indices = selectIndices(points, world)
    const mask = new Uint8Array(points.length)
    for (const i of indices) mask[i] = 1
    setSelection({ mask, indices, world })
    setSelVersion(v => v + 1)
  }

  const toggleSelecting = () => {
    draggingRef.current = false
    pathRef.current = []
    setLassoScreen([])
    setSelecting(s => !s)
  }

  const clearSelection = () => {
    setSelection(null)
    setSelVersion(v => v + 1)
  }

  const downloadCurrent = () => {
    if (!selection) return
    const artifact: SelectionArtifact = {
      store: STORE_ID,
      lasso_world: selection.world.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]),
      indices: selection.indices,
      barcodes: selection.indices.map(i => barcodes[i] ?? ''),
    }
    downloadSelection(artifact)
  }

  const layer = new ScatterplotLayer<Point>({
    id: 'umap-scatter',
    data: points,
    getPosition: d => d.position,
    getRadius: 0.5,
    radiusUnits: 'pixels',
    getFillColor: d => {
      const base = cat ? colorForCode(cat.codes[d.index]) : [130, 70, 255]
      if (selection) {
        // highlight selected in yellow; dim the rest to make it pop
        return selection.mask[d.index] ? [255, 240, 30, 255] : [base[0], base[1], base[2], 40]
      }
      return [base[0], base[1], base[2], 200]
    },
    updateTriggers: { getFillColor: [level, selVersion] },
    pickable: false,
  })

  if (loading) {
    return <div style={overlayStyle}>Loading UMAP coordinates…</div>
  }

  if (error) {
    return (
      <div style={{ ...overlayStyle, color: '#f44' }}>
        Failed to load UMAP data: {error}
      </div>
    )
  }

  return (
    <>
      <DeckGL
        ref={deckRef}
        views={new OrthographicView({ id: 'umap' })}
        initialViewState={{ target: [0, 0, 0], zoom: 3 }}
        controller={{ dragPan: !selecting }} // free the drag gesture for the lasso
        layers={[layer]}
        style={{ width: '100%', height: '100%' }}
      />

      {/* Lasso overlay — captures pointer events only while selecting. */}
      {selecting && (
        <svg
          style={{ position: 'absolute', inset: 0, zIndex: 10, cursor: 'crosshair' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          {lassoScreen.length > 1 && (
            <polygon
              points={lassoScreen.map(p => `${p[0]},${p[1]}`).join(' ')}
              fill="rgba(255, 240, 30, 0.12)"
              stroke="rgba(255, 240, 30, 0.9)"
              strokeWidth={1.5}
            />
          )}
        </svg>
      )}

      <SelectionControls
        selecting={selecting}
        onToggle={toggleSelecting}
        count={selection?.indices.length ?? 0}
        onDownload={downloadCurrent}
        onClear={clearSelection}
      />
      <Legend level={level} onLevelChange={setLevel} categories={cat?.categories ?? null} />
    </>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#ccc',
  fontSize: 16,
  background: '#111',
}
