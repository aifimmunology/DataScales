import { useEffect, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import Legend from './Legend'
import GroupPicker from './GroupPicker'
import GenePicker from './GenePicker'
import LabelsPanel from './LabelsPanel'
import RunsPanel from './RunsPanel'
import SelectionControls from './SelectionControls'
import {
  loadUmapCoords,
  loadCategorical,
  loadBarcodes,
  loadGroups,
  loadGeneNames,
  loadGeneExpression,
  loadLabelSets,
  colorForCode,
  getStoreId,
  type Point,
  type Categorical,
  type Level,
  type Group,
  type GeneExpression,
  type LabelSetInfo,
} from '../lib/zarrData'
import { selectIndices, downloadSelection, type SelectionArtifact } from '../lib/selection'
import { deleteView, saveLabels, submitSelection } from '../lib/api'

type Sel = { mask: Uint8Array; indices: number[]; world: [number, number][] }

export default function Umap() {
  const [points, setPoints] = useState<Point[]>([])
  const [barcodes, setBarcodes] = useState<string[]>([])
  const [level, setLevel] = useState<Level>('AIFI_L1') // default color-by
  const [labelSets, setLabelSets] = useState<LabelSetInfo[]>([]) // categorical obs columns in this group
  const [cat, setCat] = useState<Categorical | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)
  const [notice, setNotice] = useState<string | null>(null) // dismissible banner (failures, relocations)

  // group ("view") state — which embedding is shown
  const [groups, setGroups] = useState<Group[]>([])
  const [group, setGroup] = useState<string>('') // active group path ('' = store root)
  const [viewState, setViewState] = useState<any>({ target: [0, 0, 0], zoom: 3 })
  const [worldRadius, setWorldRadius] = useState(0.01) // point radius in embedding units

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

  // cluster-select (legend clicks) + labeling workspace
  const [selectedCats, setSelectedCats] = useState<Set<number>>(new Set())
  const [selectedLabels, setSelectedLabels] = useState<Set<number>>(new Set()) // working-label rows toggled in
  type Labeling = { name: string; cats: string[]; counts: number[]; codes: Int16Array }
  const [labeling, setLabeling] = useState<Labeling | null>(null)
  const [pending, setPending] = useState<{ label: string; indices: number[] }[]>([])
  const [labelVersion, setLabelVersion] = useState(0) // bumps fill colors on assign
  const [savingLabels, setSavingLabels] = useState(false)
  const [savedLabels, setSavedLabels] = useState<{ name: string; labeled: number } | null>(null)

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
    setSelectedCats(new Set())
    setSelectedLabels(new Set())
    setLabeling(null) // labeling rides per-group row indices — can't cross groups
    setPending([])
    setSelVersion(v => v + 1)
    Promise.all([loadUmapCoords(group), loadBarcodes(group)])
      .then(([pts, bcs]) => {
        if (stale) return
        setPoints(pts)
        setBarcodes(bcs)
        const fit = fitView(pts)
        setViewState(fit)
        // world-space radius sized so dots render ~fitPx pixels at the fitted zoom;
        // they then scale with zoom (crisp when zoomed out, resolvable zoomed in)
        const fitPx = pts.length > 1_500_000 ? 0.5 : pts.length > 300_000 ? 0.9 : pts.length > 50_000 ? 1.4 : 1.8
        setWorldRadius(fitPx / Math.pow(2, fit.zoom))
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

  // Discover which categorical obs columns this group offers; keep the current
  // level when it still exists, else fall back (AIFI_L1 first, then whatever is).
  useEffect(() => {
    let stale = false
    loadLabelSets(group)
      .then(ls => {
        if (stale) return
        setLabelSets(ls)
        const names = ls.map(s => s.name)
        setLevel(l => (names.includes(l) ? l : names.includes('AIFI_L1') ? 'AIFI_L1' : names[0] ?? ''))
      })
      .catch(() => {
        if (!stale) setLabelSets([])
      })
    return () => {
      stale = true
    }
  }, [group])

  // Category codes reload whenever the level OR the active group changes.
  // One silent retry covers a backend blip mid-load; after that the failure is
  // surfaced in the Legend (previously it hung on "Loading…" forever).
  const [catError, setCatError] = useState<string | null>(null)
  const [catTick, setCatTick] = useState(0)
  useEffect(() => {
    let stale = false
    setCat(null)
    setCatError(null)
    setSelectedCats(new Set()) // codes are per-level; stale highlights would lie
    if (!level) return // group has no categorical columns
    const attempt = (retriesLeft: number) => {
      loadCategorical(level, group)
        .then(c => {
          if (!stale) setCat(c)
        })
        .catch(err => {
          if (stale) return
          if (retriesLeft > 0) {
            setTimeout(() => !stale && attempt(retriesLeft - 1), 1500)
            return
          }
          console.error(`Failed to load ${level}:`, err)
          setCatError(err instanceof Error ? err.message : String(err))
        })
    }
    attempt(1)
    return () => {
      stale = true
    }
  }, [level, group, catTick])

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
    setSelectedCats(new Set()) // a lasso replaces any category selection
    setSelectedLabels(new Set())
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
    setSelectedCats(new Set())
    setSelectedLabels(new Set())
    setSelVersion(v => v + 1)
  }

  // Category-select: toggle all cells whose code (in `codes`) is in the toggled
  // set. Backs both the legend rows (cat codes) and the Labels-panel rows
  // (working labelset codes); the two are mutually exclusive selections.
  const selectByCodes = (codes: ArrayLike<number>, toggled: Set<number>) => {
    if (toggled.size === 0) {
      setSelection(null)
      setSelVersion(v => v + 1)
      return
    }
    const mask = new Uint8Array(points.length)
    const indices: number[] = []
    for (let i = 0; i < codes.length; i++) {
      if (toggled.has(codes[i] as number)) {
        mask[i] = 1
        indices.push(i)
      }
    }
    setSelection({ mask, indices, world: [] })
    setSelVersion(v => v + 1)
  }

  const toggleCategory = (code: number) => {
    if (!cat) return
    const next = new Set(selectedCats)
    if (next.has(code)) next.delete(code)
    else next.add(code)
    setSelectedCats(next)
    setSelectedLabels(new Set())
    selectByCodes(cat.codes, next)
  }

  const toggleLabelCat = (k: number) => {
    if (!labeling) return
    const next = new Set(selectedLabels)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    setSelectedLabels(next)
    setSelectedCats(new Set())
    selectByCodes(labeling.codes, next)
  }

  // ---- labeling: build a labelset from selections, save as an obs categorical ----
  const startLabeling = async (name: string) => {
    setSavedLabels(null)
    const codes = new Int16Array(points.length).fill(-1)
    let cats: string[] = []
    if (labelSets.some(s => s.name === name && s.own)) {
      try {
        const existing = await loadCategorical(name, group)
        cats = existing.categories
        for (let i = 0; i < codes.length; i++) codes[i] = existing.codes[i]
      } catch {
        // labelset not materialized in this group (e.g. a view made before it) — start empty
      }
    }
    const counts = new Array(cats.length).fill(0)
    for (let i = 0; i < codes.length; i++) if (codes[i] >= 0) counts[codes[i]]++
    setLabeling({ name, cats, counts, codes })
    setPending([])
    setLabelVersion(v => v + 1)
  }

  const assignLabel = (label: string) => {
    if (!labeling || !selection) return
    const cats = labeling.cats.includes(label) ? labeling.cats : [...labeling.cats, label]
    const k = cats.indexOf(label)
    const counts = [...labeling.counts]
    while (counts.length < cats.length) counts.push(0)
    for (const i of selection.indices) {
      if (labeling.codes[i] >= 0) counts[labeling.codes[i]]--
      labeling.codes[i] = k
      counts[k]++
    }
    setLabeling({ ...labeling, cats, counts })
    setPending(p => [...p, { label, indices: selection.indices }])
    setSelection(null)
    setSelectedCats(new Set())
    setSelectedLabels(new Set())
    setSelVersion(v => v + 1)
    setLabelVersion(v => v + 1)
  }

  const saveLabeling = async () => {
    if (!labeling || pending.length === 0) return
    setSavingLabels(true)
    try {
      // barcodes make the assignment group-agnostic: the backend maps them to
      // root-store rows, so view labels land on the full store
      const assignments = pending.map(p => ({
        label: p.label,
        barcodes: p.indices.map(i => barcodes[i] ?? ''),
      }))
      const res = await saveLabels(labeling.name, assignments)
      setSavedLabels({ name: res.name, labeled: res.labeled })
      setLabeling(null)
      setPending([])
      setSelectedLabels(new Set())
      setLabelVersion(v => v + 1)
      setLabelSets(await loadLabelSets(group)) // the new column joins Color by
    } catch (err) {
      setNotice(`Label save failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setSavingLabels(false)
    }
  }

  const cancelLabeling = () => {
    setLabeling(null)
    setPending([])
    setSelectedLabels(new Set())
    setLabelVersion(v => v + 1)
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
    getRadius: exprData ? worldRadius * 1.4 : worldRadius, // gene coloring pops a bit more
    radiusUnits: 'common',
    radiusMinPixels: 0.4,
    radiusMaxPixels: 5,
    getFillColor: d => {
      const i = d.index * 3
      let base: readonly number[] = exprData
        ? [exprData.colors[i], exprData.colors[i + 1], exprData.colors[i + 2]]
        : cat
          ? colorForCode(cat.codes[d.index])
          : [130, 70, 255]
      // Millions of overlapping points need low alpha to read as density; small
      // views need near-solid dots or they look fuzzy.
      let alpha = points.length > 300_000 ? 90 : 180
      if (labeling) {
        // labeled cells show their working label; the rest keep the underlying
        // coloring, dimmed, so unlabeled structure stays visible while labeling
        const wc = labeling.codes[d.index]
        if (wc >= 0) {
          base = colorForCode(wc)
          alpha = 235
        } else {
          alpha = Math.min(alpha, 55)
        }
      }
      if (selection) {
        // highlight selected in yellow; dim the rest to make it pop
        return selection.mask[d.index] ? [255, 240, 30, 255] : [base[0], base[1], base[2], 40]
      }
      return [base[0], base[1], base[2], alpha]
    },
    // `cat` MUST be here: it loads async, and deck.gl only re-runs getFillColor when
    // a trigger changes. Without it, colors stay at the default until some *other*
    // trigger fires (e.g. a lasso bumping selVersion) — which is exactly the bug where
    // color-by looked dead until you selected a subset.
    updateTriggers: { getFillColor: [level, selVersion, cat, exprData, labelVersion] },
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
        <LabelsPanel
          active={labeling ? { name: labeling.name, cats: labeling.cats, counts: labeling.counts } : null}
          ownSets={labelSets.filter(s => s.own).map(s => s.name)}
          selectionCount={selection?.indices.length ?? 0}
          saving={savingLabels}
          saved={savedLabels}
          dirty={pending.length > 0}
          selectedLabels={selectedLabels}
          onLabelClick={toggleLabelCat}
          onStart={startLabeling}
          onAssign={assignLabel}
          onSave={saveLabeling}
          onCancel={cancelLabeling}
          onDismissSaved={() => setSavedLabels(null)}
        />
        <GroupPicker groups={groups} active={group} onChange={setGroup} onDelete={deleteCurrentView} />
        <GenePicker genes={genes} active={gene} range={exprData?.range ?? null} error={exprError} warning={exprData?.warning ?? null} onChange={setGene} />
      </div>
      <RunsPanel refresh={submitCount} onViewReady={onViewReady} />
      {/* legend tracks the actual coloring mode, not the picked gene */}
      {!exprData && (
        <Legend
          levels={labelSets.map(s => s.name)}
          level={level}
          onLevelChange={setLevel}
          categories={cat?.categories ?? null}
          selected={selectedCats}
          onCategoryClick={toggleCategory}
          error={catError}
          onRetry={() => setCatTick(t => t + 1)}
        />
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
