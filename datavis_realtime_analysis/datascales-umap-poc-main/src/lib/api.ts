import type { SelectionArtifact } from './selection'

export type Job = {
  id: string
  cells: number
  group: string
  submitted_at: string
  status: 'running' | 'done' | 'failed'
}

// Submit omits barcodes: the backend derives them from the store when needed.
export async function submitSelection(artifact: Omit<SelectionArtifact, 'barcodes'>): Promise<void> {
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
