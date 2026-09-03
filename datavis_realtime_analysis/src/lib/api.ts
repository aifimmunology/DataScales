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

// Shared fetch: non-2xx throws with the server's `detail` when it sent one.
async function request(url: string, init: RequestInit | undefined, what: string): Promise<Response> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    throw new Error(detail ?? `${what} failed: ${res.status}`)
  }
  return res
}

const postJson = (url: string, body: unknown, what: string) =>
  request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, what)

// Barcodes ride along: the GPU pipeline selects cells by barcode (view-safe).
export async function submitSelection(artifact: SelectionArtifact & { name: string }): Promise<void> {
  await postJson('/api/submit', artifact, 'submit')
}

export async function fetchJobs(): Promise<Job[]> {
  const res = await fetch('/api/jobs')
  if (!res.ok) return []
  return res.json()
}

export async function deleteView(id: string): Promise<void> {
  await request(`/api/views/${encodeURIComponent(id)}`, { method: 'DELETE' }, 'delete view')
}

export type LabelAssignment = { label: string; barcodes: string[] }

// Assignments are barcode-based so labels made inside a view land on the root store.
// `seed` names an obs column the server copies as the baseline when the labelset is
// first created (fork-to-edit) — seeded labels never ride the payload.
export async function saveLabels(
  name: string,
  assignments: LabelAssignment[],
  seed?: string,
): Promise<{ name: string; categories: string[]; labeled: number }> {
  const res = await postJson('/api/labels', { name, assignments, ...(seed ? { seed } : {}) }, 'label save')
  return res.json()
}
