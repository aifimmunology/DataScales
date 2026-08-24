import { useEffect, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import Legend from './Legend'
import GroupPicker from './GroupPicker'
import GenePicker from './GenePicker'
import RunsPanel from './RunsPanel'
import SelectionControls from './SelectionControls'
import {
  loadUmapCoords,
  loadCategorical,
  loadBarcodes,
  loadGroups,
  loadGeneNames,
  loadGeneExpression,
  colorForCode,
  getStoreId,
  type Point,
  type Categorical,
  type Level,
  type Group,
  type GeneExpression,
} from '../lib/zarrData'
import { selectIndices, downloadSelection, type SelectionArtifact } from '../lib/selection'
import { deleteView, submitSelection } from '../lib/api'

type Sel = { mask: Uint8Array; indices: number[]; world: [number, number][] }

export default function Umap() {
  const [points, setPoints] = useState<Point[]>([])
  const [barcodes, setBarcodes] = useState<string[]>([])
  const [level, setLevel] = useState<Level>('AIFI_L1') // default color-by
  const [cat, setCat] = useState<Categorical | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [notice, setNotice] = useState<string | null>(null) // dismissible banner (failures, relocations)

  // group ("view") state — which embedding is shown
  const [groups, setGroups] = useState<Group[]>([])
  const [group, setGroup] = useState<string>('') // active group path ('' = store root)
  const [viewState, setViewState] = useState<any>({ target: [0, 0, 0], zoom: 3 })

  // gene expression color-by
  const [genes, setGenes] = useState<string[]>([])
  const [gene, setGene] = useState<string | null>(null)
  const [exprData, setExprData] = useState<GeneExpression | null>(null)
  const [exprError, setExprError] = useState<string | null>(null)

  // submitted-runs refresh counter (bumped on each submit)
  const [submitCount, setSubmitCount] = useState(0)

  // selection state
  const [selecting, setSelecting] = useState(false)
  const [selection, setSelection] = useState<Sel | null>(null)
  const [selVersion, setSelVersion] = useState(0) // bumps deck's getFillColor updateTrigger
  const [lassoScreen, setLassoScreen] = useState<[number, number][]>([])

  const deckRef = useRef<any>(null)
  const draggingRef = useRef(false)
  const pathRef = useRef<[number, number][]>([])

  // Discover switchable groups once; default to the first (usually the store root).
  useEffect(() => {
    loadGroups()
      .then(gs => {
        setGroups(gs)
        setGroup(gs[0]?.path ?? '')
      })
      .catch(err => console.error('Failed to load groups:', err))
  }, [])

  // Coordinates + barcodes (re)load whenever the active group changes. Row indices
  // are per-group, so drop any selection and re-fit the camera to the new extent.
  // The stale flag stops a slow response from a previous group landing after a
  // switch — mixed-group points/barcodes would corrupt a selection artifact.
  const [retryTick, setRetryTick] = useState(0)
  useEffect(() => {
    let stale = false
    setLoading(true)
    setError(null)
    setSelection(null)
    setSelVersion(v => v + 1)
    Promise.all([loadUmapCoords(group), loadBarcodes(group)])
      .then(([pts, bcs]) => {
        if (stale) return
        setPoints(pts)
        setBarcodes(bcs)
        setViewState(fitView(pts))
        setLoading(false)
      })
      .catch(async err => {
        if (stale) return
        // a view that 404s was likely deleted: re-check the listing, fall back to root
        if (group) {
          const gs = await loadGroups()
          if (stale) return
          setGroups(gs)
          if (!gs.some(g => g.path === group)) {
            setGroup('')
            setNotice('That view no longer exists — returned to the full store.')
            return
          }
        }
        console.error('Failed to load UMAP data:', err)
        setError(String(err))
        setLoading(false)
      })
    return () => {
      stale = true
    }
  }, [group, retryTick])

  // Category codes reload whenever the AIFI level OR the active group changes.
  useEffect(() => {
    let stale = false
    setCat(null)
    loadCategorical(level, group)
      .then(c => {
        if (!stale) setCat(c)
      })
      .catch(err => console.error(`Failed to load ${level}:`, err))
    return () => {
      stale = true
    }
  }, [level, group])

  // Gene list is per-group; switching groups clears the active gene.
  useEffect(() => {
    let stale = false
    setGene(null)
    loadGeneNames(group)
      .then(gs => {
        if (!stale) setGenes(gs)
      })
      .catch(() => {
        if (!stale) setGenes([])
      })
    return () => {
      stale = true
    }
  }, [group])

  useEffect(() => {
    setExprData(null)
    setExprError(null)
    const idx = gene ? genes.indexOf(gene) : -1
    if (idx < 0) return
    let stale = false
    loadGeneExpression(idx, group)
      .then(d => {
        if (!stale) setExprData(d)
      })
      .catch(err => {
        if (stale) return
        console.error(`Failed to load expression for ${gene}:`, err)
        setExprError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      stale = true
    }
  }, [gene, genes, group])

  // ---- lasso drawing (screen space) -> selection (world space) ----
  // Handlers live on a <div> overlay, not the <svg>: an empty svg defaults to
  // pointer-events:visiblePainted, so the initial pointerdown (before any polygon
  // is painted) falls through to the deck canvas and the lasso never starts. A div
  // captures over its whole box unconditionally.
  const relPos = (e: React.PointerEvent<HTMLDivElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true
    pathRef.current = [relPos(e)]
    setLassoScreen(pathRef.current.slice())
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
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
    if (!vp) {
      console.warn('[lasso] no deck viewport available — selection skipped')
      return
    }
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

  const selectionPayload = async (sel: Sel) => ({
    store: await getStoreId(),
    group, // which embedding this selection's row indices refer to
    lasso_world: sel.world.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)] as [number, number]),
    indices: sel.indices,
  })

  const downloadCurrent = async () => {
    if (!selection) return
    const artifact: SelectionArtifact = {
      ...(await selectionPayload(selection)),
      barcodes: selection.indices.map(i => barcodes[i] ?? ''),
    }
    downloadSelection(artifact)
  }

  const submitCurrent = async (name: string) => {
    if (!selection) return
    try {
      await submitSelection({
        ...(await selectionPayload(selection)),
        barcodes: selection.indices.map(i => barcodes[i] ?? ''),
        name,
      })
      setSubmitCount(c => c + 1)
      // submitted: drop lasso mode and the highlighted subset
      setSelecting(false)
      setSelection(null)
      setSelVersion(v => v + 1)
    } catch (err) {
      console.error('Submit failed:', err)
      setNotice(`Submit failed: ${err instanceof Error ? err.message : err}`)
    }
  }

  // a finished GPU run registered a new view in groups.json: reload the picker;
  // the runs panel marks it ready — no automatic switch
  const onViewReady = () => {
    loadGroups().then(setGroups)
  }

  // Relocate to the full store immediately, then delete behind the scrim; the
  // groups list refreshes either way so a failed delete stays visible in the picker.
  const deleteCurrentView = async (id: string) => {
    if (!window.confirm('Delete this view from the store? This cannot be undone.')) return
    setDeleting(true)
    setNotice(null)
    setGroup('')
    try {
      await deleteView(id)
    } catch (err) {
      console.error('Delete view failed:', err)
      setNotice(`Delete failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setGroups(await loadGroups())
      setDeleting(false)
    }
  }

  const layer = new ScatterplotLayer<Point>({
    id: 'umap-scatter',
    data: points,
    getPosition: d => d.position,
    getRadius: 0.5,
    radiusUnits: 'pixels',
    getFillColor: d => {
      const i = d.index * 3
      const base: readonly number[] = exprData
        ? [exprData.colors[i], exprData.colors[i + 1], exprData.colors[i + 2]]
        : cat
          ? colorForCode(cat.codes[d.index])
          : [130, 70, 255]
      if (selection) {
        // highlight selected in yellow; dim the rest to make it pop
        return selection.mask[d.index] ? [255, 240, 30, 255] : [base[0], base[1], base[2], 40]
      }
      // Low alpha so millions of overlapping points read as density instead of
      // saturating into one solid blob (this store is ~2.7M cells).
      return [base[0], base[1], base[2], 90]
    },
    // `cat` MUST be here: it loads async, and deck.gl only re-runs getFillColor when
    // a trigger changes. Without it, colors stay at the default until some *other*
    // trigger fires (e.g. a lasso bumping selVersion) — which is exactly the bug where
    // color-by looked dead until you selected a subset.
    updateTriggers: { getFillColor: [level, selVersion, cat, exprData] },
    pickable: false,
  })

  // Full-screen overlay only for the very first load; keep the UI (incl. the group
  // picker) mounted during a group switch so the dropdown doesn't vanish.
  if (loading && points.length === 0) {
    return <div style={overlayStyle}>Loading UMAP coordinates…</div>
  }

  if (error) {
    return (
      <div style={{ ...overlayStyle, color: '#f44', flexDirection: 'column', gap: 12 }}>
        <div>Failed to load UMAP data: {error}</div>
        <button
          onClick={() => setRetryTick(t => t + 1)}
          style={{ background: '#1c1c1c', color: '#ddd', border: '1px solid #444',
                   borderRadius: 4, padding: '6px 14px', fontSize: 13, cursor: 'pointer' }}
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <>
      <DeckGL
        ref={deckRef}
        views={new OrthographicView({ id: 'umap' })}
        viewState={viewState}
        onViewStateChange={e => setViewState(e.viewState)}
        controller={{ dragPan: !selecting }} // free the drag gesture for the lasso
        layers={[layer]}
        style={{ width: '100%', height: '100%' }}
      />

      {/* Lasso overlay — a div captures the pointer (reliable full-box hit-testing);
          the inner svg only draws the polygon (pointer-events:none so it never
          intercepts). Present only while selecting. */}
      {selecting && (
        <div
          style={{ position: 'absolute', inset: 0, zIndex: 10, cursor: 'crosshair' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            {lassoScreen.length > 1 && (
              <polygon
                points={lassoScreen.map(p => `${p[0]},${p[1]}`).join(' ')}
                fill="rgba(255, 240, 30, 0.12)"
                stroke="rgba(255, 240, 30, 0.9)"
                strokeWidth={1.5}
              />
            )}
          </svg>
        </div>
      )}

      {/* Top-left stack: selection tool on top, View picker beneath it. */}
      <div style={topLeftStack}>
        <SelectionControls
          selecting={selecting}
          onToggle={toggleSelecting}
          count={selection?.indices.length ?? 0}
          onDownload={downloadCurrent}
          onSubmit={submitCurrent}
          onClear={clearSelection}
        />
        <GroupPicker groups={groups} active={group} onChange={setGroup} onDelete={deleteCurrentView} />
        <GenePicker genes={genes} active={gene} range={exprData?.range ?? null} error={exprError} warning={exprData?.warning ?? null} onChange={setGene} />
      </div>
      <RunsPanel refresh={submitCount} onViewReady={onViewReady} />
      {/* legend tracks the actual coloring mode, not the picked gene */}
      {!exprData && (
        <Legend level={level} onLevelChange={setLevel} categories={cat?.categories ?? null} />
      )}

      {notice && (
        <div style={noticeStyle} onClick={() => setNotice(null)} title="dismiss">
          {notice}
        </div>
      )}
      {/* pointer-blocking scrim: also prevents a second ✕ click mid-delete */}
      {deleting && (
        <div style={{ ...overlayStyle, background: 'rgba(17,17,17,0.75)', zIndex: 40 }}>
          Deleting view…
        </div>
      )}
    </>
  )
}

// Fit the orthographic camera to a set of points: center on the bbox and pick a zoom
// that fits the larger extent into the viewport (OrthographicView: pixels = world *
// 2^zoom). Each embedding lives in its own coordinate range, so we refit on every
// group switch instead of assuming a fixed target/zoom.
function fitView(pts: Point[]): { target: [number, number, number]; zoom: number } {
  if (pts.length === 0) return { target: [0, 0, 0], zoom: 3 }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of pts) {
    const [x, y] = p.position
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX)) return { target: [0, 0, 0], zoom: 3 }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
  const extentX = Math.max(maxX - minX, 1e-6), extentY = Math.max(maxY - minY, 1e-6)
  const vw = window.innerWidth || 800, vh = window.innerHeight || 600
  const zoom = Math.log2(Math.min(vw / extentX, vh / extentY) * 0.9)
  return { target: [cx, cy, 0], zoom: Math.max(-2, Math.min(zoom, 10)) }
}

// Top-left overlay column: selection controls, then the View picker below them.
// zIndex 20 keeps it above the lasso svg overlay (zIndex 10).
const topLeftStack: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  left: 12,
  zIndex: 20,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  alignItems: 'flex-start',
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

const noticeStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 30,
  background: '#2a2a18',
  color: '#ffe94d',
  border: '1px solid #665c1e',
  borderRadius: 4,
  padding: '6px 12px',
  fontSize: 13,
  cursor: 'pointer',
}
