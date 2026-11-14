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

/** `n` is the group's cell count: the backend refuses a selection made against a stale copy. */
export type Selection = { group: string; n: number; indices: number[] }

type LabelAssignment = { label: string; indices: number[] }

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

export async function submitSelection(body: Selection & { name: string; params: RunParams }): Promise<void> {
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

export async function saveLabels(
  name: string,
  assignments: LabelAssignment[],
  group: string,
  n: number,
): Promise<{ name: string; categories: string[]; labeled: number }> {
  return (await postJson('/api/labels', { name, group, n, assignments }, 'label save')).json()
}
