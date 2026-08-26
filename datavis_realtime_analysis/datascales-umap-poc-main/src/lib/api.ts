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
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    throw new Error(detail ?? `submit failed: ${res.status}`)
  }
}

export async function fetchJobs(): Promise<Job[]> {
  const res = await fetch('/api/jobs')
  if (!res.ok) return []
  return res.json()
}

export async function deleteView(id: string): Promise<void> {
  const res = await fetch(`/api/views/${encodeURIComponent(id)}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`delete view failed: ${res.status}`)
}

export type LabelAssignment = { label: string; barcodes: string[] }

// Assignments are barcode-based so labels made inside a view land on the root store.
export async function saveLabels(
  name: string,
  assignments: LabelAssignment[],
): Promise<{ name: string; categories: string[]; labeled: number }> {
  const res = await fetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, assignments }),
  })
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    throw new Error(detail ?? `label save failed: ${res.status}`)
  }
  return res.json()
}
