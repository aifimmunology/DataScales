import { useEffect, useRef, useState } from 'react'
import { fetchJobs, type GpuHealth, type Job } from '../lib/api'
import { describeParams } from '../lib/runSettings'
import { ghostBtn, muted, panel } from '../lib/styles'

const STATUS: Record<Job['status'], { icon: string; color: string }> = {
  queued: { icon: '…', color: '#888' },
  running: { icon: '⟳', color: '#ffe94d' },
  done: { icon: '✓', color: '#7f7' },
  failed: { icon: '✗', color: '#f88' },
}

const POLL_MS = 2000

const isActive = (j: Job) => j.status === 'running' || j.status === 'queued'

const fmtElapsed = (iso: string) => {
  const s = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
}

type Props = {
  refresh: number
  onViewReady: (path: string) => void
  onActiveChange: (active: boolean) => void
  health: GpuHealth | null
  onRecheckHealth: (refresh?: boolean) => void
}

export default function RunsPanel({ refresh, onViewReady, onActiveChange, health, onRecheckHealth }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [, setTick] = useState(0)
  const notified = useRef(new Set<string>())
  const poll = () => fetchJobs().then(setJobs).catch(() => {})

  useEffect(() => {
    poll()
  }, [refresh])

  const active = jobs.some(isActive)
  useEffect(() => onActiveChange(active), [active])
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => {
      poll()
      setTick(v => v + 1)
    }, POLL_MS)
    return () => clearInterval(t)
  }, [active])

  useEffect(() => {
    for (const j of jobs) {
      if (j.status === 'done' && j.view && !notified.current.has(j.id)) {
        notified.current.add(j.id)
        onViewReady(j.view)
      }
    }
  }, [jobs])

  const failedCount = jobs.filter(j => j.status === 'failed').length
  useEffect(() => {
    if (failedCount > 0) onRecheckHealth()
  }, [failedCount])

  return (
    <>
      <HealthNote health={health} onRecheck={() => onRecheckHealth(true)} />
      {jobs.length === 0 ? (
        <div style={muted}>No GPU runs yet this session.</div>
      ) : (
        <div style={listStyle}>
          {jobs.map(j => (
            <div key={j.id} style={rowStyle}>
              <span style={{ color: STATUS[j.status].color }}>{STATUS[j.status].icon}</span>
              <span style={{ color: '#ddd' }}>{j.name || `#${j.id}`}</span>
              <span>{j.cells.toLocaleString()} cells</span>
              {j.params && <span title="run parameters">— {describeParams(j.params)}</span>}
              {isActive(j) && <span style={{ color: '#ffe94d' }}>{fmtElapsed(j.submitted_at)}</span>}
              {j.status === 'running' && j.stage && <span>— {j.stage}</span>}
              {j.status === 'done' && j.view && <span style={{ color: '#7f7' }}>— view ready in the View picker</span>}
              {j.status === 'failed' && j.stage && (
                <span style={{ color: '#f88' }} title={j.stage}>
                  — {j.stage.slice(0, 60)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  )
}

function HealthNote({ health, onRecheck }: { health: GpuHealth | null; onRecheck: () => void }) {
  if (!health) return null
  if (health.status === 'unconfigured') return <div style={muted}>{health.summary}</div>
  if (health.status === 'checking') return <div style={muted}>Checking GPU access…</div>
  if (health.status === 'ok') {
    return (
      <div style={muted}>
        <span style={{ color: '#7f7' }}>✓</span> GPU access verified
        {health.checking ? (
          <span> — re-checking…</span>
        ) : (
          <button onClick={onRecheck} style={{ ...ghostBtn, marginLeft: 8 }}>
            re-check
          </button>
        )}
      </div>
    )
  }
  return (
    <div style={errBannerStyle}>
      <div style={{ color: '#f88', fontWeight: 600 }}>⚠ GPU access problem</div>
      <div>{health.summary}</div>
      {health.detail && <div style={detailStyle}>{health.detail}</div>}
      {!!health.fix?.length && (
        <div>
          <div style={{ color: '#888', margin: '4px 0 2px' }}>How to fix:</div>
          <pre style={fixStyle}>{health.fix.join('\n')}</pre>
        </div>
      )}
      <button onClick={onRecheck} style={ghostBtn} disabled={health.checking}>
        {health.checking ? 'checking…' : 'Re-check'}
      </button>
    </div>
  )
}

const listStyle: React.CSSProperties = { ...panel, fontSize: 12, color: '#999' }

const rowStyle: React.CSSProperties = { display: 'flex', gap: 8, alignItems: 'baseline', padding: '2px 0', flexWrap: 'wrap' }

const errBannerStyle: React.CSSProperties = {
  ...panel,
  fontSize: 12,
  color: '#ddd',
  border: '1px solid rgba(255, 136, 136, 0.5)',
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
}

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }
const detailStyle: React.CSSProperties = { ...mono, color: '#999' }
const fixStyle: React.CSSProperties = { ...mono, color: '#bcd8ff', background: '#0d0d0d', padding: '6px 8px', borderRadius: 6, margin: 0 }
