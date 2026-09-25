import { useState } from 'react'
import { panel, control, label as labelStyle, primaryBtn, rgb } from '../lib/styles'
import { colorForCode, type LabelSetInfo } from '../lib/zarrData'

type Active = { name: string; cats: string[]; counts: number[] }

type Props = {
  active: Active | null
  columns: LabelSetInfo[] // every categorical obs column in this group
  selectionCount: number
  saving: boolean
  saved: { name: string; labeled: number } | null // last completed save
  dirty: boolean
  selectedLabels: Set<number> // label rows currently cluster-selected
  onLabelClick: (k: number) => void
  onStart: (name: string, seed?: string) => void
  onAssign: (label: string) => void
  onSave: () => void
  onCancel: () => void
  onDismissSaved: () => void
}

// Rail section: build a labelset from selections (lasso or legend cluster
// clicks), then save it to the store as a categorical obs column. The search
// covers every categorical obs column: own sets extend in place, anything else
// forks into a new editable labelset seeded from it.
export default function LabelsPanel({
  active,
  columns,
  selectionCount,
  saving,
  saved,
  dirty,
  selectedLabels,
  onLabelClick,
  onStart,
  onAssign,
  onSave,
  onCancel,
  onDismissSaved,
}: Props) {
  const [query, setQuery] = useState('')
  const [seed, setSeed] = useState<string | null>(null) // column picked to fork from
  const [saveAs, setSaveAs] = useState('')
  const [lab, setLab] = useState('')

  const taken = (n: string) => columns.some(c => c.name === n)

  if (!active) {
    if (seed) {
      const name = saveAs.trim()
      return (
        <div style={panelStyle}>
          <label style={labelStyle}>Labels — fork '{seed}'</label>
          <input
            value={saveAs}
            placeholder="save new labelset as…"
            onChange={e => setSaveAs(e.target.value)}
            style={inputStyle}
          />
          {name && taken(name) && (
            <div style={{ fontSize: 11, color: '#f88' }}>that obs column already exists</div>
          )}
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              disabled={!name || taken(name)}
              onClick={() => {
                onStart(name, seed)
                setSeed(null)
                setSaveAs('')
                setQuery('')
              }}
              style={btnStyle}
            >
              ✎ Start from '{seed}'
            </button>
            <button onClick={() => setSeed(null)} style={{ ...btnStyle, color: '#888' }}>
              back
            </button>
          </div>
        </div>
      )
    }
    const q = query.trim()
    const matches = q
      ? columns.filter(c => c.name.toLowerCase().includes(q.toLowerCase())).slice(0, 20)
      : []
    return (
      <div style={panelStyle}>
        <label style={labelStyle}>Labels</label>
        {saved && (
          <div style={savedStyle} onClick={onDismissSaved} title="dismiss">
            ✓ '{saved.name}' saved — {saved.labeled.toLocaleString()} cells labeled
          </div>
        )}
        <input
          value={query}
          placeholder={`Search ${columns.length} obs columns, or name a new set…`}
          onChange={e => setQuery(e.target.value)}
          style={inputStyle}
        />
        {matches.length > 0 && (
          <div style={listStyle}>
            {matches.map(c => (
              <div
                key={c.name}
                style={itemStyle}
                title={c.own ? 'app labelset — extend it' : 'store column — fork an editable copy'}
                onClick={() => {
                  if (c.own) {
                    onStart(c.name)
                    setQuery('')
                  } else {
                    setSeed(c.name)
                    setSaveAs(`${c.name}_edited`)
                  }
                }}
              >
                {c.name}{' '}
                <span style={{ color: '#888' }}>
                  ({c.nCats}) · {c.own ? 'extend' : 'fork'}
                </span>
              </div>
            ))}
          </div>
        )}
        {q && !taken(q) && (
          <button onClick={() => { onStart(q); setQuery('') }} style={btnStyle}>
            ＋ New empty labelset '{q}'
          </button>
        )}
      </div>
    )
  }

  return (
    <div style={panelStyle}>
      <label style={labelStyle}>Labeling: {active.name}</label>
      {active.cats.map((c, k) => (
        <div
          key={c}
          onClick={() => onLabelClick(k)}
          title="click to select these cells"
          style={{
            display: 'flex',
            gap: 6,
            alignItems: 'center',
            fontSize: 12,
            cursor: 'pointer',
            padding: '1px 4px',
            borderRadius: 3,
            background: selectedLabels.has(k) ? '#3a3411' : 'transparent',
          }}
        >
          <span style={{ width: 10, height: 10, borderRadius: 2, background: rgb(colorForCode(k)), flexShrink: 0 }} />
          <span style={{ color: selectedLabels.has(k) ? '#ffe94d' : '#ddd' }}>{c}</span>
          <span style={{ color: '#888' }}>{active.counts[k]?.toLocaleString()}</span>
        </div>
      ))}
      <input
        value={lab}
        placeholder="label for selection…"
        onChange={e => setLab(e.target.value)}
        style={inputStyle}
      />
      <button
        disabled={!lab.trim() || selectionCount === 0}
        onClick={() => {
          onAssign(lab.trim())
          setLab('')
        }}
        style={btnStyle}
      >
        Assign to {selectionCount.toLocaleString()} cells
      </button>
      {saving ? (
        <div>
          <div style={barTrackStyle}>
            <div style={barFillStyle} />
          </div>
          <div style={{ fontSize: 11, color: '#ffe94d', marginTop: 4 }}>Saving to store…</div>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 6 }}>
          <button disabled={!dirty} onClick={onSave} style={dirty ? primaryBtn : btnStyle}>
            💾 Save to store
          </button>
          <button onClick={onCancel} style={{ ...btnStyle, color: '#f88', borderColor: '#622' }}>
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  ...panel,
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  maxWidth: 220,
}

const inputStyle: React.CSSProperties = { ...control, cursor: 'text' }

const btnStyle: React.CSSProperties = {
  ...control,
  padding: '6px 10px',
  cursor: 'pointer',
  textAlign: 'left',
}

const savedStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#7f7',
  cursor: 'pointer',
}

const listStyle: React.CSSProperties = {
  maxHeight: 180,
  overflowY: 'auto',
  border: '1px solid #333',
  borderRadius: 4,
  background: '#1c1c1c',
}

const itemStyle: React.CSSProperties = {
  padding: '4px 8px',
  fontSize: 13,
  cursor: 'pointer',
}

const barTrackStyle: React.CSSProperties = {
  position: 'relative',
  height: 6,
  background: '#222',
  border: '1px solid #333',
  borderRadius: 3,
  overflow: 'hidden',
}

const barFillStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '40%',
  height: '100%',
  background: 'linear-gradient(90deg, #6a5, #ffe94d)',
  animation: 'datavis-bar 1.2s linear infinite',
}
