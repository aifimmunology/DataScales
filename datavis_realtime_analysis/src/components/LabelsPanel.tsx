import { useState } from 'react'
import { panel, control, label as labelStyle, primaryBtn, rgb } from '../lib/styles'
import { colorForCode, type LabelSetInfo } from '../lib/zarrData'
import { validLabelsetName } from '../lib/labeling'
import ComboBox from './ComboBox'

type Active = { name: string; cats: string[]; counts: number[] }

type Props = {
  active: Active | null
  columns: LabelSetInfo[] // every categorical obs column in this group
  labelFor: (name: string) => string
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

// Rail section: pick any categorical obs column (or name a new one), relabel
// selections (lasso or legend cluster clicks), then save it back in place.
export default function LabelsPanel({
  active,
  columns,
  labelFor,
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
        <ComboBox
          options={columns.map(c => ({ value: c.name, label: labelFor(c.name), hint: `(${c.nCats})` }))}
          placeholder={`Edit one of ${columns.length} columns, or name a new set…`}
          onPick={onStart}
          createLabel={q => (columns.some(c => c.name === q) || !validLabelsetName(q) ? null : `＋ New empty labelset '${q}'`)}
          onCreate={onStart}
        />
      </div>
    )
  }

  const canAssign = !!lab.trim() && selectionCount > 0 && !saving
  return (
    <div style={panelStyle}>
      <label style={labelStyle}>Labeling: {labelFor(active.name)}</label>
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
        disabled={!canAssign}
        onClick={() => {
          onAssign(lab.trim())
          setLab('')
        }}
        style={canAssign ? primaryBtn : offStyle}
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
          <button disabled={!dirty} onClick={onSave} style={!dirty ? offStyle : canAssign ? btnStyle : primaryBtn}>
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

const offStyle: React.CSSProperties = { ...btnStyle, opacity: 0.4, cursor: 'not-allowed' }

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
