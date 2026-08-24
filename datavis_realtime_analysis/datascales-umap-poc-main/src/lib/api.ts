import type { SelectionArtifact } from './selection'

export type Job = {
  id: string
  name?: string
  cells: number
  group: string
  submitted_at: string
  status: 'queued' | 'running' | 'done' | 'failed'
  stage?: string
  view?: string | null
}

// Barcodes ride along: the GPU pipeline selects cells by barcode (view-safe).
export async function submitSelection(artifact: SelectionArtifact & { name: string }): Promise<void> {
  const res = await fetch('/api/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(artifact),
  })
  if (!res.ok) throw new Error(`submit failed: ${res.status}`)
}

export async function fetchJobs(): Promise<Job[]> {
  const res = await fetch('/api/jobs')
  if (!res.ok) return []
  return res.json()
}
