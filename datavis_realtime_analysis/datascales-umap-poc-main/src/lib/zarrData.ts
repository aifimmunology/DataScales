import * as zarr from 'zarrita'

// All zarr reads go through the backend's /api/data proxy (Vite in dev, nginx in Docker).
export const DATA_BASE_URL = window.location.origin + '/api/data'

// The backend's DATA_DIR — recorded as provenance in selection exports.
let storeIdPromise: Promise<string> | null = null
export function getStoreId(): Promise<string> {
  storeIdPromise ??= fetch('/api/config')
    .then(res => (res.ok ? res.json() : Promise.reject(res.status)))
    .then((cfg: { store?: string }) => cfg.store || DATA_BASE_URL)
    .catch(() => DATA_BASE_URL)
  return storeIdPromise
}

// AIFI cell-type annotation levels, coarse -> fine. L1 is the default color-by.
export const LEVELS = ['AIFI_L1', 'AIFI_L2', 'AIFI_L3'] as const
export type Level = (typeof LEVELS)[number]

export type RGB = [number, number, number]

// Categorical palette (tab20-ish, colorblind-leaning). Indexed by category code;
// wraps if a level has more categories than colors.
const PALETTE: RGB[] = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
]

export function colorForCode(code: number): RGB {
  if (code < 0) return [130, 130, 130] // NaN / unassigned
  return PALETTE[code % PALETTE.length]
}

// Expression ramp: dim gray (low) -> bright yellow (high), t in [0, 1].
export function exprColor(t: number): RGB {
  return [60 + t * 195, 60 + t * 160, 70 - t * 10]
}

// A "group" is either the served store root (path '') or a nested view store
// (e.g. 'umap_views/bcell_selection'). Every array fetch is prefixed by it, so the
// same obsm/X_umap + obs/* paths resolve inside whichever group is active.
async function openArray(path: string, group = '') {
  const rel = group ? `${group}/${path}` : path
  const store = new zarr.FetchStore(`${DATA_BASE_URL}/${rel}`)
  return zarr.open.v3(store, { kind: 'array' })
}

export type Point = { position: [number, number]; index: number }

/** obsm/X_umap -> one Point per cell, for the given group. */
export async function loadUmapCoords(group = ''): Promise<Point[]> {
  const arr = await openArray('obsm/X_umap', group)
  const data = (await zarr.get(arr)).data as Float32Array | Float64Array
  const n = arr.shape[0]
  const points: Point[] = new Array(n)
  for (let i = 0; i < n; i++) {
    points[i] = { position: [data[i * 2], data[i * 2 + 1]], index: i }
  }
  return points
}

/** obs/barcodes -> one id per cell (obs_names). Used for the selection export. */
export async function loadBarcodes(group = ''): Promise<string[]> {
  const arr = await openArray('obs/barcodes', group)
  const data = (await zarr.get(arr)).data as ArrayLike<string>
  return Array.from(data)
}

export type Categorical = { codes: Int8Array | Int16Array | Int32Array; categories: string[] }

/** An AnnData categorical obs column: per-cell integer `codes` + `categories`. */
export async function loadCategorical(col: Level, group = ''): Promise<Categorical> {
  const codesArr = await openArray(`obs/${col}/codes`, group)
  const codes = (await zarr.get(codesArr)).data as Int8Array | Int16Array | Int32Array
  const catsArr = await openArray(`obs/${col}/categories`, group)
  const categories = Array.from((await zarr.get(catsArr)).data as ArrayLike<string>)
  return { codes, categories }
}

/** Gene names from the var index column (named per the group's `_index` attr). */
export async function loadGeneNames(group = ''): Promise<string[]> {
  const rel = group ? `${group}/var` : 'var'
  const varGroup = await zarr.open.v3(new zarr.FetchStore(`${DATA_BASE_URL}/${rel}`), { kind: 'group' })
  const idxCol = (varGroup.attrs['_index'] as string) ?? '_index'
  const arr = await openArray(`var/${idxCol}`, group)
  return Array.from((await zarr.get(arr)).data as ArrayLike<string>)
}

export type GeneExpression = { colors: Uint8Array; range: [number, number] }

// A one-column read fetches every chunk intersecting the column; refuse layouts
// where that means streaming a large share of X (row-oriented chunks).
const MAX_COLUMN_READ_BYTES = 256 * 1024 * 1024

/** One X column (dense stores only) -> per-cell ramp colors + value range. */
export async function loadGeneExpression(geneIdx: number, group = ''): Promise<GeneExpression> {
  const arr = await openArray('X', group)
  const n = arr.shape[0]
  const readBytes = n * arr.chunks[1] * 4
  if (readBytes > MAX_COLUMN_READ_BYTES) {
    throw new Error(
      `store not laid out for gene queries: one gene read would fetch ~${Math.round(readBytes / 1e6)} MB`,
    )
  }
  const vals = (await zarr.get(arr, [null, geneIdx])).data as ArrayLike<number>
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    const v = vals[i]
    if (v < min) min = v
    if (v > max) max = v
  }
  const span = max - min || 1
  const colors = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) {
    const [r, g, b] = exprColor((vals[i] - min) / span)
    colors[i * 3] = r
    colors[i * 3 + 1] = g
    colors[i * 3 + 2] = b
  }
  return { colors, range: [min, max] }
}

// ── Groups: switchable embeddings (the root store + any nested view stores) ─────
export type Group = { id: string; label: string; path: string }

// Fallback when there's no groups.json: a single group = the served store root,
// so the app behaves exactly as it did before this feature existed.
const DEFAULT_GROUPS: Group[] = [{ id: 'main', label: 'Full store', path: '' }]

/** groups.json at the served root lists the switchable embeddings. Missing or
 * invalid -> a single root group. */
export async function loadGroups(): Promise<Group[]> {
  try {
    const res = await fetch(`${DATA_BASE_URL}/groups.json`)
    if (!res.ok) return DEFAULT_GROUPS
    const gs = (await res.json()) as Group[]
    return Array.isArray(gs) && gs.length > 0 ? gs : DEFAULT_GROUPS
  } catch {
    return DEFAULT_GROUPS
  }
}
