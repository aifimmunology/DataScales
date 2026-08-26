import { useEffect, useRef, useState } from 'react'
import { fetchJobs, type Job } from '../lib/api'
import { panel } from '../lib/styles'

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

type Props = { refresh: number; onViewReady: (path: string) => void }

// refresh bumps on submit; polling runs while a job is queued/running; when a job
// finishes with a view path, onViewReady fires once so the View picker can refresh.
export default function RunsPanel({ refresh, onViewReady }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const notified = useRef(new Set<string>())
  const poll = () => fetchJobs().then(setJobs).catch(() => {})

  useEffect(() => {
    poll()
  }, [refresh])

  const active = jobs.some(j => j.status === 'running' || j.status === 'queued')
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

  if (jobs.length === 0) return null

  return (
    <div style={panelStyle}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>GPU runs</div>
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
  )
}

const panelStyle: React.CSSProperties = {
  ...panel,
  position: 'absolute',
  bottom: 12,
  left: 12,
  zIndex: 20,
  fontSize: 12,
  color: '#999',
  maxHeight: 200,
  maxWidth: 520,
  overflowY: 'auto',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  alignItems: 'baseline',
  padding: '2px 0',
  whiteSpace: 'nowrap',
}
