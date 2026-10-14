// Per-run GPU pipeline settings: what the settings drawer edits, what /api/submit
// carries (`RunParams`, mirroring gpu/run_params.py), and how a view's recorded
// parameters are summarised.

export type Method = 'pca' | 'latent'

export type RunSettings = {
  method: Method | null // null = follow the store's default route
  nNeighbors: number
  minDist: number
  resolution: number
  batchKey: string | null
}

export type EmbeddingInfo = {
  latent: string | null
  default: Method
  defaults: { n_neighbors: number; min_dist: number; resolution: number }
}

export type RunParams = {
  method: Method
  n_neighbors: number
  min_dist: number
  resolution: number
  batch_key: string | null
}

export type RecordedParams = Partial<RunParams> & { rep?: string | null }

const FALLBACK = { n_neighbors: 20, min_dist: 0.45, resolution: 1.1 }
const STORAGE_KEY = 'datavis.runSettings.v1'

export function defaultSettings(info: EmbeddingInfo | null): RunSettings {
  const d = info?.defaults ?? FALLBACK
  return { method: null, nNeighbors: d.n_neighbors, minDist: d.min_dist, resolution: d.resolution, batchKey: null }
}

export function effectiveMethod(s: RunSettings, info: EmbeddingInfo | null): Method {
  const wanted = s.method ?? info?.default ?? 'pca'
  return wanted === 'latent' && !info?.latent ? 'pca' : wanted
}

export function toParams(s: RunSettings, info: EmbeddingInfo | null): RunParams {
  const method = effectiveMethod(s, info)
  return {
    method,
    n_neighbors: s.nNeighbors,
    min_dist: s.minDist,
    resolution: s.resolution,
    batch_key: method === 'pca' ? s.batchKey : null,
  }
}

export function loadSettings(info: EmbeddingInfo | null): RunSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...defaultSettings(info), ...JSON.parse(raw) } : defaultSettings(info)
  } catch {
    return defaultSettings(info)
  }
}

export function saveSettings(s: RunSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
  } catch {
    /* private window or blocked storage: settings just don't persist */
  }
}

export function describeParams(p: RecordedParams | null | undefined): string | null {
  if (!p) return null
  const route = p.rep ? (p.rep.startsWith('X_pca') ? 'PCA' : p.rep) : p.method === 'latent' ? 'latent' : 'PCA'
  const parts = [route, `k ${p.n_neighbors}`, `min_dist ${p.min_dist}`, `res ${p.resolution}`]
  if (p.batch_key) parts.push(`harmony ${p.batch_key}`)
  return parts.join(' · ')
}
