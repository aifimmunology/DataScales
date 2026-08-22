import * as zarr from 'zarrita'

// All zarr reads go through the backend's /api/data proxy (Vite in dev, nginx in Docker).
export const DATA_BASE_URL = window.location.origin + '/api/data'

// Store recorded in selection exports/submissions: the rapids store when the backend
// was given a separate one (RAPIDS_DIR), else the vis store itself.
let storeIdPromise: Promise<string> | null = null
export function getStoreId(): Promise<string> {
  storeIdPromise ??= fetch('/api/config')
    .then(res => (res.ok ? res.json() : Promise.reject(res.status)))
    .then((cfg: { store?: string; rapids_store?: string }) =>
      cfg.rapids_store || cfg.store || DATA_BASE_URL)
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

// A dense one-column read fetches every chunk intersecting the column; refuse
// layouts where that means streaming a large share of X (row-oriented chunks).
const MAX_COLUMN_READ_BYTES = 256 * 1024 * 1024

type GeneSource =
  | { kind: 'dense'; path: string }
  | { kind: 'csc'; path: string; nObs: number }

const geneSources = new Map<string, Promise<GeneSource>>()
const cscIndptrs = new Map<string, Promise<number[]>>()

function openAt(path: string) {
  return zarr.open.v3(new zarr.FetchStore(`${DATA_BASE_URL}/${path}`), { kind: 'array' })
}

async function probe(path: string) {
  const res = await fetch(`${DATA_BASE_URL}/${path}/zarr.json`)
  if (!res.ok) return null
  const meta = await res.json()
  return { node: meta.node_type as string, attrs: (meta.attributes ?? {}) as Record<string, unknown> }
}

// Gene reads resolve to layers/gexp first (the zarrsmith setup), then X itself;
// dense arrays and CSC groups are readable, CSR is not (a column read would scan it all).
async function resolveGeneSource(group: string): Promise<GeneSource> {
  const prefix = group ? `${group}/` : ''
  for (const path of [`${prefix}layers/gexp`, `${prefix}X`]) {
    const p = await probe(path)
    if (!p) continue
    if (p.node === 'array') return { kind: 'dense', path }
    if (p.attrs['encoding-type'] === 'csc_matrix') {
      const [nObs] = p.attrs['shape'] as [number, number]
      return { kind: 'csc', path, nObs }
    }
  }
  throw new Error(
    'no gene-readable matrix (need layers/gexp or a dense/CSC X) — run `zarrsmith add-expr --format csc` on the store',
  )
}

function geneSource(group: string): Promise<GeneSource> {
  let src = geneSources.get(group)
  if (!src) {
    src = resolveGeneSource(group)
    geneSources.set(group, src)
  }
  return src
}

// CSC column = two tiny range reads: data/indices[indptr[j]:indptr[j+1]], scattered
// into a dense per-cell vector. indptr is read once per source and cached.
async function readCscColumn(src: { path: string; nObs: number }, geneIdx: number): Promise<Float32Array> {
  let indptr = cscIndptrs.get(src.path)
  if (!indptr) {
    indptr = openAt(`${src.path}/indptr`)
      .then(arr => zarr.get(arr))
      .then(r => Array.from(r.data as ArrayLike<number | bigint>, Number))
    cscIndptrs.set(src.path, indptr)
  }
  const ip = await indptr
  const out = new Float32Array(src.nObs)
  const [start, end] = [ip[geneIdx], ip[geneIdx + 1]]
  if (start === end) return out
  const sel = [zarr.slice(start, end)]
  const [vals, rows] = await Promise.all([
    openAt(`${src.path}/data`).then(a => zarr.get(a, sel)),
    openAt(`${src.path}/indices`).then(a => zarr.get(a, sel)),
  ])
  const v = vals.data as ArrayLike<number>
  const r = rows.data as ArrayLike<number | bigint>
  for (let i = 0; i < v.length; i++) out[Number(r[i])] = Number(v[i])
  return out
}

async function readDenseColumn(path: string, geneIdx: number): Promise<ArrayLike<number>> {
  const arr = await openAt(path)
  const readBytes = arr.shape[0] * arr.chunks[1] * 4
  if (readBytes > MAX_COLUMN_READ_BYTES) {
    throw new Error(
      `store not laid out for gene queries: one gene read would fetch ~${Math.round(readBytes / 1e6)} MB`,
    )
  }
  return (await zarr.get(arr, [null, geneIdx])).data as ArrayLike<number>
}

/** One gene column (layers/gexp, dense X, or CSC X) -> per-cell ramp colors + value range. */
export async function loadGeneExpression(geneIdx: number, group = ''): Promise<GeneExpression> {
  const src = await geneSource(group)
  const vals = src.kind === 'csc'
    ? await readCscColumn(src, geneIdx)
    : await readDenseColumn(src.path, geneIdx)
  const n = vals.length
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < n; i++) {
    const v = vals[i] as number
    if (v < min) min = v
    if (v > max) max = v
  }
  const span = max - min || 1
  const colors = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) {
    const [r, g, b] = exprColor(((vals[i] as number) - min) / span)
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
