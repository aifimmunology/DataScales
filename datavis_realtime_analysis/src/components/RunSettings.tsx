import { useEffect, useState, type ReactNode } from 'react'
import { DIVIDER, control, label, panel } from '../lib/styles'
import { defaultSettings, effectiveMethod, type EmbeddingInfo, type Method, type RunSettings } from '../lib/runSettings'

type Props = {
  settings: RunSettings
  onChange: (s: RunSettings) => void
  embedding: EmbeddingInfo | null
  batchKeys: string[]
}

// Secondary drawer beside the Selection panel: the GPU pipeline parameters applied
// to the next "Generate New UMAP". Pre-filled from the store's defaults.
export default function RunSettingsPanel({ settings, onChange, embedding, batchKeys }: Props) {
  const method = effectiveMethod(settings, embedding)
  const latent = embedding?.latent
  const set = (patch: Partial<RunSettings>) => onChange({ ...settings, ...patch })
  return (
    <div style={panelStyle}>
      <Field name="Embedding">
        <select value={method} onChange={e => set({ method: e.target.value as Method })} style={selectStyle}>
          <option value="pca">Normalize → PCA from counts</option>
          <option value="latent" disabled={!latent}>
            {latent ? `Stored latent space (${latent})` : 'Stored latent space (none in this store)'}
          </option>
          <option value="refit" disabled>
            Re-fit scVI / PeakVI on the subset (coming soon)
          </option>
        </select>
      </Field>
      <NumberField name="Neighbors (k)" value={settings.nNeighbors} min={2} step={1} onCommit={v => set({ nNeighbors: Math.round(v) })} />
      <NumberField name="UMAP min_dist" value={settings.minDist} min={0} max={1} step={0.05} onCommit={v => set({ minDist: v })} />
      <NumberField name="Leiden resolution" value={settings.resolution} min={0.05} step={0.1} onCommit={v => set({ resolution: v })} />
      <Field name="Harmony batch key" hint={method === 'latent' ? 'PCA route only' : undefined}>
        <select
          value={settings.batchKey ?? ''}
          disabled={method === 'latent'}
          onChange={e => set({ batchKey: e.target.value || null })}
          style={selectStyle}
        >
          <option value="">none</option>
          {batchKeys.map(k => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </Field>
      <button onClick={() => onChange(defaultSettings(embedding))} style={resetStyle}>
        Reset to defaults
      </button>
    </div>
  )
}

function Field({ name, hint, children }: { name: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <label style={label}>
        {name}
        {hint && <span style={{ color: '#666' }}> — {hint}</span>}
      </label>
      {children}
    </div>
  )
}

type NumberProps = { name: string; value: number; min?: number; max?: number; step: number; onCommit: (v: number) => void }

// Edits a draft string so partial input ("0.") is allowed; commits on blur/Enter when valid.
function NumberField({ name, value, min, max, step, onCommit }: NumberProps) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])
  const commit = () => {
    const v = Number(draft)
    const ok = draft !== '' && Number.isFinite(v) && (min === undefined || v >= min) && (max === undefined || v <= max)
    if (ok) onCommit(v)
    else setDraft(String(value))
  }
  return (
    <Field name={name}>
      <input
        type="number"
        value={draft}
        min={min}
        max={max}
        step={step}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => e.key === 'Enter' && commit()}
        style={inputStyle}
      />
    </Field>
  )
}

const panelStyle: React.CSSProperties = { ...panel, display: 'flex', flexDirection: 'column', gap: 10 }
const inputStyle: React.CSSProperties = { ...control, width: '100%', boxSizing: 'border-box' }
const selectStyle: React.CSSProperties = { ...control, width: '100%' }
const resetStyle: React.CSSProperties = {
  alignSelf: 'flex-start',
  background: 'none',
  border: `1px solid ${DIVIDER}`,
  color: '#aaa',
  borderRadius: 6,
  padding: '3px 10px',
  fontSize: 12,
  cursor: 'pointer',
}
