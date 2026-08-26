import { useState } from 'react'
import { panel, control, label as labelStyle, rgb } from '../lib/styles'
import { colorForCode } from '../lib/zarrData'

type Active = { name: string; cats: string[]; counts: number[] }

type Props = {
  active: Active | null
  ownSets: string[] // existing app-created labelsets (extendable)
  selectionCount: number
  saving: boolean
  saved: { name: string; labeled: number } | null // last completed save
  dirty: boolean
  selectedLabels: Set<number> // label rows currently cluster-selected
  onLabelClick: (k: number) => void
  onStart: (name: string) => void
  onAssign: (label: string) => void
  onSave: () => void
  onCancel: () => void
  onDismissSaved: () => void
}

// Top-left overlay: build a labelset from selections (lasso or legend cluster
// clicks), then save it to the store as a categorical obs column.
export default function LabelsPanel({
  active,
  ownSets,
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
  const [name, setName] = useState('')
  const [lab, setLab] = useState('')

  if (!active) {
    return (
      <div style={panelStyle}>
        <label style={labelStyle}>Labels</label>
        {saved && (
          <div style={savedStyle} onClick={onDismissSaved} title="dismiss">
            ✓ '{saved.name}' saved — {saved.labeled.toLocaleString()} cells labeled
          </div>
        )}
        <input
          value={name}
          placeholder="new or existing labelset…"
          onChange={e => setName(e.target.value)}
          style={inputStyle}
          list="own-labelsets"
        />
        <datalist id="own-labelsets">
          {ownSets.map(s => (
            <option key={s} value={s} />
          ))}
        </datalist>
        <button
          disabled={!name.trim()}
          onClick={() => {
            onStart(name.trim())
            setName('')
          }}
          style={btnStyle}
        >
          ✎ Start labeling
        </button>
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
          <button disabled={!dirty} onClick={onSave} style={btnStyle}>
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
