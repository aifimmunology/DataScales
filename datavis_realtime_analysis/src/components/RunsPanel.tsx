import { useEffect, useRef, useState } from 'react'
import { fetchJobs, type GpuHealth, type Job } from '../lib/api'
import { DIVIDER, panel } from '../lib/styles'

const STATUS: Record<Job['status'], { icon: string; color: string }> = {
  queued: { icon: '…', color: '#888' },
  running: { icon: '⟳', color: '#ffe94d' },
  done: { icon: '✓', color: '#7f7' },
  failed: { icon: '✗', color: '#f88' },
}

const fmtElapsed = (iso: string) => {
  const s = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
}

type Props = {
  refresh: number
  onViewReady: (path: string) => void
  onActiveChange?: (active: boolean) => void // drives the side-rail badge
  health: GpuHealth | null // GPU access probe, owned by Umap (probed on app load)
  onRecheckHealth: (refresh?: boolean) => void
}

// refresh bumps on submit; polling runs while a job is queued/running; when a job
// finishes with a view path, onViewReady fires once so the View picker can refresh.
export default function RunsPanel({ refresh, onViewReady, onActiveChange, health, onRecheckHealth }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const notified = useRef(new Set<string>())
  const poll = () => fetchJobs().then(setJobs).catch(() => {})

  useEffect(() => {
    poll()
  }, [refresh])

  const active = jobs.some(j => j.status === 'running' || j.status === 'queued')
  useEffect(() => {
    onActiveChange?.(active)
  }, [active, onActiveChange])
  useEffect(() => {
    if (!active) return
    const t = setInterval(poll, 2000)
    return () => clearInterval(t)
  }, [active])

  // 1s display tick so the elapsed timer runs smoothly while a job is active
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setTick(v => v + 1), 1000)
    return () => clearInterval(t)
  }, [active])

  useEffect(() => {
    for (const j of jobs) {
      if (j.status === 'done' && j.view && !notified.current.has(j.id)) {
        notified.current.add(j.id)
        onViewReady(j.view)
      }
    }
  }, [jobs, onViewReady])

  // a failed job is often an expired credential — re-probe so the banner explains it
  // (the backend invalidates its probe cache on job failure, so no forced refresh needed)
  const failedCount = jobs.filter(j => j.status === 'failed').length
  useEffect(() => {
    if (failedCount > 0) onRecheckHealth()
  }, [failedCount])

  return (
    <>
      <HealthNote health={health} onRecheck={() => onRecheckHealth(true)} />
      {jobs.length === 0 ? (
        <div style={{ fontSize: 12, color: '#888' }}>No GPU runs yet this session.</div>
      ) : (
        <div style={panelStyle}>
          {jobs.map(j => (
            <div key={j.id} style={rowStyle}>
              <span style={{ color: STATUS[j.status].color }}>{STATUS[j.status].icon}</span>
              <span style={{ color: '#ddd' }}>{j.name || `#${j.id}`}</span>
              <span>{j.cells.toLocaleString()} cells</span>
              {(j.status === 'running' || j.status === 'queued') && (
                <span style={{ color: '#ffe94d' }}>{fmtElapsed(j.submitted_at)}</span>
              )}
              {j.status === 'running' && j.stage && <span>— {j.stage}</span>}
              {j.status === 'done' && j.view && (
                <span style={{ color: '#7f7' }}>— view ready in the View picker</span>
              )}
              {j.status === 'failed' && j.stage && (
                <span style={{ color: '#f88' }} title={j.stage}>— {j.stage.slice(0, 60)}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  )
}

// Temp while GPU dispatch rides ssh — surfaces the backend's permission probe
// with fix steps so an expired credential is visible before a run dies.
function HealthNote({ health, onRecheck }: { health: GpuHealth | null; onRecheck: () => void }) {
  if (!health) return null
  if (health.status === 'unconfigured') {
    return <div style={mutedStyle}>{health.summary}</div>
  }
  if (health.status === 'checking') {
    return <div style={mutedStyle}>Checking GPU access…</div>
  }
  if (health.status === 'ok') {
    return (
      <div style={mutedStyle}>
        <span style={{ color: '#7f7' }}>✓</span> GPU access verified
        {health.checking ? (
          <span> — re-checking…</span>
        ) : (
          <button onClick={onRecheck} style={{ ...recheckBtnStyle, marginLeft: 8 }}>
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
      {health.fix && health.fix.length > 0 && (
        <div>
          <div style={{ color: '#888', margin: '4px 0 2px' }}>How to fix:</div>
          <pre style={fixStyle}>{health.fix.join('\n')}</pre>
        </div>
      )}
      <button onClick={onRecheck} style={recheckBtnStyle} disabled={health.checking}>
        {health.checking ? 'checking…' : 'Re-check'}
      </button>
    </div>
  )
}

const panelStyle: React.CSSProperties = {
  ...panel,
  fontSize: 12,
  color: '#999',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  alignItems: 'baseline',
  padding: '2px 0',
  flexWrap: 'wrap',
}

const mutedStyle: React.CSSProperties = { fontSize: 12, color: '#888' }

const errBannerStyle: React.CSSProperties = {
  ...panel,
  fontSize: 12,
  color: '#ddd',
  border: '1px solid rgba(255, 136, 136, 0.5)',
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
}

const detailStyle: React.CSSProperties = {
  fontFamily: 'monospace',
  fontSize: 11,
  color: '#999',
  whiteSpace: 'pre-wrap',
  overflowWrap: 'anywhere',
}

const fixStyle: React.CSSProperties = {
  fontFamily: 'monospace',
  fontSize: 11,
  color: '#bcd8ff',
  background: '#0d0d0d',
  padding: '6px 8px',
  borderRadius: 6,
  margin: 0,
  whiteSpace: 'pre-wrap',
  overflowWrap: 'anywhere',
}

const recheckBtnStyle: React.CSSProperties = {
  alignSelf: 'flex-start',
  background: 'none',
  border: `1px solid ${DIVIDER}`,
  color: '#aaa',
  borderRadius: 6,
  padding: '2px 8px',
  fontSize: 11,
  cursor: 'pointer',
}
