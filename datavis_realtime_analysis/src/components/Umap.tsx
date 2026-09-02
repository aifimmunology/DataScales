import { useEffect, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import Header, { HEADER_HEIGHT } from './Header'
import Legend from './Legend'
import GroupPicker from './GroupPicker'
import GenePicker from './GenePicker'
import LabelsPanel from './LabelsPanel'
import RunsPanel from './RunsPanel'
import SelectionControls from './SelectionControls'
import SideRail from './SideRail'
import {
  loadUmapCoords,
  loadCategorical,
  gatherRootCategorical,
  loadBarcodes,
  loadGroups,
  loadGeneNames,
  loadGeneExpression,
  loadLabelSets,
  dropdownLabelSets,
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
  const [runsActive, setRunsActive] = useState(false) // any job queued/running → rail badge

  // selection state
  const [selecting, setSelecting] = useState(false)
  const [selection, setSelection] = useState<Sel | null>(null)
  const [selVersion, setSelVersion] = useState(0) // bumps deck's getFillColor updateTrigger
  const [lassoScreen, setLassoScreen] = useState<[number, number][]>([])

  // cluster-select (legend clicks) + labeling workspace
  const [selectedCats, setSelectedCats] = useState<Set<string>>(new Set()) // category names, not codes — survives view switches
  const [selectedLabels, setSelectedLabels] = useState<Set<number>>(new Set()) // working-label rows toggled in
  type Labeling = { name: string; cats: string[]; counts: number[]; codes: Int16Array; seed?: string }
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
        // Dot radius ∝ 1/√n (dot AREA ∝ 1/n, scanpy's law): total ink stays
        // constant across dataset sizes instead of jumping between tiers.
        // ≈0.4px at 3M (density regime), ≈0.9px at 300k, ≈2.2px at 50k,
        // ≈5px at 10k, capped at 6px for tiny views. Sized at the fitted zoom,
        // then scales with zoom (crisp zoomed out, resolvable zoomed in).
        const fitPx = Math.min(6, Math.max(0.4, 500 / Math.sqrt(pts.length || 1)))
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

  // Discover the group's categorical columns AND the root's: app labelsets live
  // only on root (edited by barcode, read in views via root_row gather), so views
  // must see them regardless of what got frozen into their own obs at creation.
  const [rootLabelSets, setRootLabelSets] = useState<LabelSetInfo[]>([])
  const rootOwn = rootLabelSets.filter(s => s.own)
  useEffect(() => {
    let stale = false
    Promise.all([loadLabelSets(group), group ? loadLabelSets('') : null])
      .then(([ls, rootLs]) => {
        if (stale) return
        setLabelSets(ls)
        setRootLabelSets(rootLs ?? ls)
        // legend options: the group's columns plus root labelsets (root wins on
        // name clashes — a view's copy is frozen at creation time)
        const ownNames = (rootLs ?? ls).filter(s => s.own).map(s => s.name)
        const names = [...new Set([...dropdownLabelSets(ls), ...ownNames])]
        setLevel(l => (names.includes(l) ? l : names.includes('AIFI_L1') ? 'AIFI_L1' : names[0] ?? ''))
      })
      .catch(() => {
        if (!stale) {
          setLabelSets([])
          setRootLabelSets([])
        }
      })
    return () => {
      stale = true
    }
  }, [group])

  // new labelset = new categories; group switches keep the selection
  useEffect(() => {
    setSelectedCats(new Set())
  }, [level])

  // Category codes reload whenever the level OR the active group changes.
  // One silent retry covers a backend blip mid-load; after that the failure is
  // surfaced in the Legend (previously it hung on "Loading…" forever).
  const [catError, setCatError] = useState<string | null>(null)
  const [catTick, setCatTick] = useState(0)
  useEffect(() => {
    let stale = false
    setCat(null)
    setCatError(null)
    if (!level) return // group has no categorical columns
    // app labelsets in a view read the CURRENT root state via root_row gather
    const fromRoot = group !== '' && rootOwn.some(s => s.name === level)
    const attempt = (retriesLeft: number) => {
      ;(fromRoot ? gatherRootCategorical(level, group) : loadCategorical(level, group))
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
  }, [level, group, catTick, rootLabelSets])

  // Gene names come from the root store (views gather root expression via
  // obs/root_row); switching groups just clears the active gene.
  useEffect(() => {
    let stale = false
    setGene(null)
    loadGeneNames()
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

  const codesFor = (c: Categorical, names: Set<string>) => {
    const out = new Set<number>()
    c.categories.forEach((name, k) => {
      if (names.has(name)) out.add(k)
    })
    return out
  }

  const toggleCategory = (code: number) => {
    if (!cat) return
    const name = cat.categories[code]
    const next = new Set(selectedCats)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    setSelectedCats(next)
    setSelectedLabels(new Set())
    selectByCodes(cat.codes, codesFor(cat, next))
  }

  // reapply the kept legend selection once the new group's cat + points both land
  useEffect(() => {
    if (!cat || selectedCats.size === 0) return
    if (cat.codes.length !== points.length) return
    selectByCodes(cat.codes, codesFor(cat, selectedCats))
  }, [cat, points])

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
  // seedFrom forks any ROOT categorical into a new labelset: local codes seed from
  // it here; at save the backend copies the same column as the stored baseline.
  // Labeling always operates in root space — in a view, seeds are gathered from
  // the current root column via root_row (never the view's frozen copy).
  const startLabeling = async (name: string, seedFrom?: string) => {
    setSavedLabels(null)
    const codes = new Int16Array(points.length).fill(-1)
    let cats: string[] = []
    const src = seedFrom ?? (rootOwn.some(s => s.name === name) ? name : undefined)
    if (src) {
      try {
        const existing = group ? await gatherRootCategorical(src, group) : await loadCategorical(src, '')
        cats = existing.categories
        for (let i = 0; i < codes.length; i++) codes[i] = existing.codes[i]
      } catch {
        // seed column unreadable (e.g. a pre-root_row view) — start empty
      }
    }
    const counts = new Array(cats.length).fill(0)
    for (let i = 0; i < codes.length; i++) if (codes[i] >= 0) counts[codes[i]]++
    setLabeling({ name, cats, counts, codes, seed: seedFrom })
    setPending([])
    setSelectedLabels(new Set())
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
      const res = await saveLabels(labeling.name, assignments, labeling.seed)
      setSavedLabels({ name: res.name, labeled: res.labeled })
      setLabeling(null)
      setPending([])
      setSelectedLabels(new Set())
      setLabelVersion(v => v + 1)
      // the saved set lives on root: refresh both lists so it joins Color by here too
      const [ls, rootLs] = await Promise.all([loadLabelSets(group), group ? loadLabelSets('') : null])
      setLabelSets(ls)
      setRootLabelSets(rootLs ?? ls)
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
      setSelecting(false)
      // a submitted lasso is done; label-driven selections stay
      if (selectedCats.size === 0 && selectedLabels.size === 0) {
        setSelection(null)
        setSelVersion(v => v + 1)
      }
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
    radiusMaxPixels: 8, // let small views' dots grow when zoomed in for cell-level inspection
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

  // Side-rail sections: the app's feature panels, spatial-explorer style. The
  // Legend stays floating on the canvas — it's the plot key and the
  // cluster-click surface, so it can't live behind a rail click.
  const railSections = [
    {
      id: 'select',
      icon: 'material-symbols-light:lasso-select',
      title: 'Selection',
      badge: (selection?.indices.length ?? 0) > 0,
      content: (
        <SelectionControls
          selecting={selecting}
          onToggle={toggleSelecting}
          count={selection?.indices.length ?? 0}
          onDownload={downloadCurrent}
          onSubmit={submitCurrent}
          onClear={clearSelection}
        />
      ),
    },
    {
      id: 'labels',
      icon: 'material-symbols-light:new-label-outline',
      title: 'Labels',
      badge: !!labeling,
      content: (
        <LabelsPanel
          active={labeling ? { name: labeling.name, cats: labeling.cats, counts: labeling.counts } : null}
          columns={rootLabelSets /* labeling operates in root space, even inside views */}
          selectionCount={selection?.indices.length ?? 0}
          saving={savingLabels}
          saved={savedLabels}
          dirty={pending.length > 0 || !!labeling?.seed}
          selectedLabels={selectedLabels}
          onLabelClick={toggleLabelCat}
          onStart={startLabeling}
          onAssign={assignLabel}
          onSave={saveLabeling}
          onCancel={cancelLabeling}
          onDismissSaved={() => setSavedLabels(null)}
        />
      ),
    },
    {
      id: 'views',
      icon: 'material-symbols-light:scatter-plot-outline',
      title: 'Views',
      content:
        groups.length > 1 ? (
          <GroupPicker groups={groups} active={group} onChange={setGroup} onDelete={deleteCurrentView} />
        ) : (
          <span style={mutedStyle}>No saved views yet — lasso a selection and run it on the GPU.</span>
        ),
    },
    {
      id: 'genes',
      icon: 'material-symbols-light:genetics',
      title: 'Genes',
      badge: !!gene,
      content:
        genes.length > 0 ? (
          <GenePicker genes={genes} active={gene} range={exprData?.range ?? null} error={exprError} warning={exprData?.warning ?? null} onChange={setGene} />
        ) : (
          <span style={mutedStyle}>No gene-readable matrix in this store.</span>
        ),
    },
    {
      id: 'runs',
      icon: 'material-symbols-light:memory',
      title: 'GPU runs',
      badge: runsActive,
      content: <RunsPanel refresh={submitCount} onViewReady={onViewReady} onActiveChange={setRunsActive} />,
    },
  ]

  const canvas =
    loading && points.length === 0 ? (
      <div style={overlayStyle}>Loading UMAP coordinates…</div>
    ) : error ? (
      <div style={{ ...overlayStyle, color: '#f44', flexDirection: 'column', gap: 12 }}>
        <div>Failed to load UMAP data: {error}</div>
        <button
          onClick={() => setRetryTick(t => t + 1)}
          style={{ background: '#1c1c1c', color: '#ddd', border: '1px solid #444',
                   borderRadius: 6, padding: '6px 14px', fontSize: 13, cursor: 'pointer' }}
        >
          Retry
        </button>
      </div>
    ) : (
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

        {/* legend tracks the actual coloring mode, not the picked gene */}
        {!exprData && (
          <Legend
            levels={[...new Set([...dropdownLabelSets(labelSets), ...rootOwn.map(s => s.name)])]}
            level={level}
            onLevelChange={setLevel}
            categories={cat?.categories ?? null}
            selected={cat ? codesFor(cat, selectedCats) : new Set<number>()}
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
      </>
    )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Header />
      <div style={{ display: 'flex', flex: 1, minHeight: 0, position: 'relative' }}>
        <SideRail sections={railSections} />
        <div style={{ flex: 1, position: 'relative', minWidth: 0, background: '#111' }}>
          {canvas}
        </div>
        {/* pointer-blocking scrim: also prevents a second ✕ click mid-delete */}
        {deleting && (
          <div style={{ ...overlayStyle, background: 'rgba(17,17,17,0.75)', zIndex: 40 }}>
            Deleting view…
          </div>
        )}
      </div>
    </div>
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
  // canvas = viewport minus the header (top) and rail (left)
  const vw = (window.innerWidth || 800) - HEADER_HEIGHT, vh = (window.innerHeight || 600) - HEADER_HEIGHT
  const zoom = Math.log2(Math.min(vw / extentX, vh / extentY) * 0.9)
  return { target: [cx, cy, 0], zoom: Math.max(-2, Math.min(zoom, 10)) }
}

const mutedStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#888',
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
