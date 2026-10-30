import { useState } from 'react'
import { button, buttonOff, column, control, dangerButton, label as labelStyle, primaryBtn, rgb } from '../lib/styles'
import type { LabelSetInfo } from '../lib/columns'
import { validLabelsetName } from '../lib/labeling'
import { colorForCode } from '../lib/palette'
import type { Saved } from '../hooks/useLabeling'
import ComboBox from './ComboBox'

type Props = {
  active: { name: string; cats: string[]; counts: number[] } | null
  columns: LabelSetInfo[]
  labelFor: (name: string) => string
  selectionCount: number
  saving: boolean
  saved: Saved | null
  dirty: boolean
  selectedLabels: Set<number>
  onLabelClick: (k: number) => void
  onStart: (name: string) => void
  onAssign: (label: string) => void
  onSave: () => void
  onCancel: () => void
  onDismissSaved: () => void
}

export default function LabelsPanel(p: Props) {
  const [lab, setLab] = useState('')

  if (!p.active) {
    return (
      <div style={column(6)}>
        <label style={labelStyle}>Labels</label>
        {p.saved && (
          <div style={savedStyle} onClick={p.onDismissSaved} title="dismiss">
            ✓ '{p.saved.name}' saved — {p.saved.labeled.toLocaleString()} cells labeled
          </div>
        )}
        <ComboBox
          options={p.columns.map(c => ({ value: c.name, label: p.labelFor(c.name), hint: `(${c.nCats})` }))}
          placeholder={`Edit one of ${p.columns.length} columns, or name a new set…`}
          onPick={p.onStart}
          createLabel={q => (p.columns.some(c => c.name === q) || !validLabelsetName(q) ? null : `＋ New empty labelset '${q}'`)}
          onCreate={p.onStart}
        />
      </div>
    )
  }

  const canAssign = !!lab.trim() && p.selectionCount > 0 && !p.saving
  return (
    <div style={column(6)}>
      <label style={labelStyle}>Labeling: {p.labelFor(p.active.name)}</label>
      {p.active.cats.map((c, k) => {
        const on = p.selectedLabels.has(k)
        return (
          <div key={c} onClick={() => p.onLabelClick(k)} title="click to select these cells" style={{ ...rowStyle, background: on ? '#3a3411' : 'transparent' }}>
            <span style={{ ...swatchStyle, background: rgb(colorForCode(k)) }} />
            <span style={{ color: on ? '#ffe94d' : '#ddd' }}>{c}</span>
            <span style={{ color: '#888' }}>{p.active!.counts[k]?.toLocaleString()}</span>
          </div>
        )
      })}
      <input value={lab} placeholder="label for selection…" onChange={e => setLab(e.target.value)} style={{ ...control, cursor: 'text' }} />
      <button
        disabled={!canAssign}
        onClick={() => {
          p.onAssign(lab.trim())
          setLab('')
        }}
        style={canAssign ? primaryBtn : buttonOff}
      >
        Assign to {p.selectionCount.toLocaleString()} cells
      </button>
      {p.saving ? (
        <div>
          <div style={barTrackStyle}>
            <div style={barFillStyle} />
          </div>
          <div style={{ fontSize: 11, color: '#ffe94d', marginTop: 4 }}>Saving to store…</div>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 6 }}>
          <button disabled={!p.dirty} onClick={p.onSave} style={!p.dirty ? buttonOff : canAssign ? button : primaryBtn}>
            💾 Save to store
          </button>
          <button onClick={p.onCancel} style={dangerButton}>
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

const rowStyle: React.CSSProperties = { display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, cursor: 'pointer', padding: '1px 4px', borderRadius: 3 }
const swatchStyle: React.CSSProperties = { width: 10, height: 10, borderRadius: 2, flexShrink: 0 }
const savedStyle: React.CSSProperties = { fontSize: 12, color: '#7f7', cursor: 'pointer' }

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
