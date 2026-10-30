import { useEffect, useMemo, useState } from 'react'
import Header from './Header'
import Legend from './Legend'
import GroupPicker from './GroupPicker'
import GenePicker from './GenePicker'
import LabelsPanel from './LabelsPanel'
import RunsPanel from './RunsPanel'
import ScatterCanvas from './ScatterCanvas'
import SelectionControls from './SelectionControls'
import RunSettingsPanel from './RunSettings'
import SideRail, { type RailSection } from './SideRail'
import { submitSelection } from '../lib/api'
import { errorText } from '../lib/async'
import { columnLabel } from '../lib/columns'
import { pointColors } from '../lib/pointColors'
import { toParams } from '../lib/runSettings'
import { lassoIndices } from '../lib/selection'
import { useCategorical, useColumns } from '../hooks/useColumns'
import { useEmbedding } from '../hooks/useEmbedding'
import { useGenes } from '../hooks/useGenes'
import { useGpuHealth } from '../hooks/useGpuHealth'
import { useLabeling } from '../hooks/useLabeling'
import { useRunSettings } from '../hooks/useRunSettings'
import { useSelection } from '../hooks/useSelection'
import { useViews } from '../hooks/useViews'

const GENE_DOT_SCALE = 1.4
const MIN_BATCH_CATS = 2
const MAX_BATCH_CATS = 200

export default function Umap() {
  const [notice, setNotice] = useState<string | null>(null)
  const [retryTick, setRetryTick] = useState(0)
  const [storeVersion, setStoreVersion] = useState(0)
  const [selecting, setSelecting] = useState(false)
  const [submitCount, setSubmitCount] = useState(0)
  const [runsActive, setRunsActive] = useState(false)

  const views = useViews(setNotice)
  const { group } = views
  const emb = useEmbedding(group, retryTick)
  const n = emb.data?.n ?? 0
  const { columns, rootSets, levels, level, setLevel } = useColumns(group, storeVersion + retryTick)
  const cat = useCategorical(level, group, storeVersion + retryTick)
  const genes = useGenes(group)
  const gpu = useGpuHealth()
  const run = useRunSettings()
  const sel = useSelection(n, group, level, cat.data)
  const labeling = useLabeling(group, emb.data?.barcodes ?? [], setNotice, name => {
    setStoreVersion(v => v + 1)
    if (name) setLevel(name)
  })

  useEffect(() => {
    if (emb.error && group) views.recheck(group)
  }, [emb.error])

  const expr = genes.expr.data
  const colors = useMemo(
    () =>
      emb.data &&
      pointColors(emb.data.order, {
        codes: cat.data?.codes,
        expr: expr?.colors,
        labels: labeling.shown?.codes,
        mask: sel.selection?.mask,
      }),
    [emb.data, cat.data, expr, labeling.shown, sel.selection],
  )
  const hasUnlabeled = useMemo(() => !!cat.data?.codes.some(c => c < 0), [cat.data])
  const labelFor = (name: string) => columnLabel(group, name)
  const selectedCount = sel.selection?.indices.length ?? 0

  const onLasso = (world: [number, number][]) => emb.data && sel.lasso(lassoIndices(emb.data.coords, world))

  const assignLabel = (label: string) => {
    if (!sel.selection) return
    labeling.assign(label, sel.selection.indices)
    sel.clear()
  }

  const submit = async (name: string) => {
    if (!sel.selection || !emb.data) return
    const barcodes = emb.data.barcodes
    try {
      await submitSelection({ barcodes: sel.selection.indices.map(i => barcodes[i]), name, params: toParams(run.settings, run.embedding) })
      setSubmitCount(c => c + 1)
      setSelecting(false)
      sel.dropLasso()
    } catch (err) {
      setNotice(`Submit failed: ${errorText(err)}`)
    }
  }

  const railSections: RailSection[] = [
    {
      id: 'select',
      icon: 'material-symbols-light:lasso-select',
      title: 'Selection',
      badge: selectedCount > 0,
      content: (
        <SelectionControls
          selecting={selecting}
          onToggle={() => setSelecting(s => !s)}
          count={selectedCount}
          onSubmit={submit}
          onClear={sel.clear}
        />
      ),
      aside: {
        icon: 'material-symbols-light:settings-outline',
        title: 'Run settings',
        content: (
          <RunSettingsPanel
            settings={run.settings}
            onChange={run.update}
            embedding={run.embedding}
            batchKeys={rootSets.filter(s => s.nCats >= MIN_BATCH_CATS && s.nCats <= MAX_BATCH_CATS).map(s => s.name)}
          />
        ),
      },
    },
    {
      id: 'labels',
      icon: 'material-symbols-light:new-label-outline',
      title: 'Labels',
      badge: !!labeling.shown,
      content: (
        <LabelsPanel
          active={labeling.shown}
          columns={columns}
          labelFor={labelFor}
          selectionCount={selectedCount}
          saving={labeling.saving}
          saved={labeling.saved}
          dirty={labeling.dirty}
          selectedLabels={sel.labelRows}
          onLabelClick={k => labeling.shown && sel.toggleLabelRow(labeling.shown.codes, k)}
          onStart={labeling.start}
          onAssign={assignLabel}
          onSave={labeling.save}
          onCancel={labeling.cancel}
          onDismissSaved={labeling.dismissSaved}
        />
      ),
    },
    {
      id: 'views',
      icon: 'material-symbols-light:scatter-plot-outline',
      title: 'Views',
      content: <GroupPicker groups={views.groups} active={group} onChange={views.setGroup} onDelete={views.remove} />,
    },
    ...(genes.genes.length > 0
      ? [{
          id: 'genes',
          icon: 'material-symbols-light:genetics',
          title: 'Genes',
          badge: !!genes.gene,
          content: (
            <GenePicker
              genes={genes.genes}
              active={genes.gene}
              range={expr?.range ?? null}
              error={genes.expr.error}
              warning={expr?.warning ?? null}
              onChange={genes.setGene}
            />
          ),
        }]
      : []),
    {
      id: 'runs',
      icon: 'material-symbols-light:memory',
      title: 'GPU runs',
      badge: runsActive || gpu.problem,
      badgeColor: gpu.problem ? '#f88' : undefined,
      content: (
        <RunsPanel
          refresh={submitCount}
          onViewReady={() => views.refresh()}
          onActiveChange={setRunsActive}
          health={gpu.health}
          onRecheckHealth={gpu.check}
        />
      ),
    },
  ]

  const canvas = emb.error ? (
    <div style={{ ...overlayStyle, color: '#f44', flexDirection: 'column', gap: 12 }}>
      <div>Failed to load UMAP data: {emb.error}</div>
      <button onClick={() => setRetryTick(t => t + 1)} style={retryStyle}>
        Retry
      </button>
    </div>
  ) : !emb.data || !colors ? (
    <div style={overlayStyle}>Loading UMAP coordinates…</div>
  ) : (
    <>
      <ScatterCanvas
        coords={emb.data.coords}
        order={emb.data.order}
        colors={colors}
        dotScale={expr ? GENE_DOT_SCALE : 1}
        selecting={selecting}
        onLasso={onLasso}
      />
      {!expr && (
        <Legend
          levels={levels}
          labelFor={labelFor}
          level={level}
          onLevelChange={setLevel}
          categories={cat.data?.categories ?? null}
          unlabeled={hasUnlabeled}
          selected={sel.legendCodes}
          onCategoryClick={sel.toggleCategory}
          error={cat.error}
          onRetry={() => setRetryTick(t => t + 1)}
        />
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
          {notice && (
            <div style={noticeStyle} onClick={() => setNotice(null)} title="dismiss">
              {notice}
            </div>
          )}
        </div>
        {views.deleting && <div style={{ ...overlayStyle, background: 'rgba(17,17,17,0.75)', zIndex: 40 }}>Deleting view…</div>}
      </div>
    </div>
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

const retryStyle: React.CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 6,
  padding: '6px 14px',
  fontSize: 13,
  cursor: 'pointer',
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
