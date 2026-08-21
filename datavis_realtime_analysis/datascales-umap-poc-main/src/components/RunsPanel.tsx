import { useEffect, useState } from 'react'
import { fetchJobs, type Job } from '../lib/api'
import { panel } from '../lib/styles'

const STATUS: Record<Job['status'], { icon: string; color: string }> = {
  running: { icon: '⟳', color: '#ffe94d' },
  done: { icon: '✓', color: '#7f7' },
  failed: { icon: '✗', color: '#f88' },
}

// refresh bumps on submit; polling runs only while a job is still running.
export default function RunsPanel({ refresh }: { refresh: number }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const poll = () => fetchJobs().then(setJobs).catch(() => {})

  useEffect(() => {
    poll()
  }, [refresh])

  const running = jobs.some(j => j.status === 'running')
  useEffect(() => {
    if (!running) return
    const t = setInterval(poll, 3000)
    return () => clearInterval(t)
  }, [running])

  if (jobs.length === 0) return null

  return (
    <div style={panelStyle}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 6 }}>Submitted runs</div>
      {jobs.map(j => (
        <div key={j.id} style={rowStyle}>
          <span style={{ color: STATUS[j.status].color }}>{STATUS[j.status].icon}</span>
          <span style={{ color: '#ddd' }}>#{j.id}</span>
          <span>{j.cells.toLocaleString()} cells</span>
          <span>{new Date(j.submitted_at).toLocaleTimeString()}</span>
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
  overflowY: 'auto',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 8,
  alignItems: 'baseline',
  padding: '2px 0',
}
