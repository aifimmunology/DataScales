import type { RecordedParams, RunParams } from './runSettings'

export type Job = {
  id: string
  name?: string
  cells: number
  submitted_at: string
  status: 'queued' | 'running' | 'done' | 'failed'
  stage?: string
  view?: string | null
  params?: RecordedParams | null
}

export type Group = { id: string; label: string; path: string; params?: RecordedParams | null }

export type GpuHealth = {
  status: 'ok' | 'error' | 'checking' | 'unconfigured'
  checking: boolean
  problem?: string | null
  summary?: string
  detail?: string
  fix?: string[]
  checked_at?: string | null
}

type LabelAssignment = { label: string; barcodes: string[] }

async function request(url: string, init: RequestInit | undefined, what: string): Promise<Response> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const detail = (await res.json().catch(() => null))?.detail
    throw new Error(detail ?? `${what} failed: ${res.status}`)
  }
  return res
}

const postJson = (url: string, body: unknown, what: string) =>
  request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, what)

export async function submitSelection(body: { barcodes: string[]; name: string; params: RunParams }): Promise<void> {
  await postJson('/api/submit', body, 'submit')
}

export async function fetchJobs(): Promise<Job[]> {
  const res = await fetch('/api/jobs')
  return res.ok ? res.json() : []
}

export async function fetchViews(): Promise<Group[]> {
  return (await request('/api/views', { cache: 'no-store' }, 'views')).json()
}

export async function fetchGpuHealth(refresh = false): Promise<GpuHealth | null> {
  const res = await fetch(`/api/gpu/health${refresh ? '?refresh=1' : ''}`)
  return res.ok ? res.json() : null
}

export async function deleteView(id: string): Promise<void> {
  await request(`/api/views/${encodeURIComponent(id)}`, { method: 'DELETE' }, 'delete view')
}

/** Barcode assignments, so labels made inside a view land on the root store; `group`
 * targets a view's own column instead. */
export async function saveLabels(
  name: string,
  assignments: LabelAssignment[],
  group?: string,
): Promise<{ name: string; categories: string[]; labeled: number }> {
  return (await postJson('/api/labels', { name, assignments, ...(group ? { group } : {}) }, 'label save')).json()
}
