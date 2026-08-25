import * as zarr from 'zarrita'

// All zarr reads go through the backend's /api/data proxy (Vite in dev, nginx in Docker).
export const DATA_BASE_URL = window.location.origin + '/api/data'

type AppConfig = { store?: string; rapids_store?: string }
let cfgPromise: Promise<AppConfig> | null = null
function getConfig(): Promise<AppConfig> {
  cfgPromise ??= fetch(window.location.origin + '/api/config')
    .then(res => (res.ok ? res.json() : Promise.reject(new Error(`config ${res.status}`))))
    .catch(() => {
      cfgPromise = null // transient failure (backend restarting): retry on next call
      return {}
    })
  return cfgPromise
}

// Store recorded in selection exports/submissions: the rapids store when the backend
// was given a separate one (RAPIDS_DIR), else the vis store itself.
export async function getStoreId(): Promise<string> {
  const cfg = await getConfig()
  return cfg.rapids_store || cfg.store || DATA_BASE_URL
}

// With two stores, coords/labels/barcodes/views come from the rapids store
// (/api/rapids-data); the vis store (/api/data) only answers gene expression.
async function coordBase(): Promise<string> {
  const cfg = await getConfig()
  return cfg.rapids_store && cfg.rapids_store !== cfg.store
    ? window.location.origin + '/api/rapids-data'
    : DATA_BASE_URL
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

// Expression ramp over EXPRESSING cells: viridis-style blue -> teal -> green ->
// yellow. Multi-hue so adjacent expression levels stay distinguishable; starts at
// blue (not viridis's near-black purple) to keep the low end distinct from EXPR_NONE.
const RAMP: RGB[] = [
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
]

export function exprColor(t: number): RGB {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1)
  const i = Math.min(Math.floor(x), RAMP.length - 2)
  const f = x - i
  const a = RAMP[i]
  const b = RAMP[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

// Zero/no expression: dark, recedes behind the ramp.
export const EXPR_NONE: RGB = [46, 46, 58]

// A "group" is either the served store root (path '') or a nested view store
// (e.g. 'umap_views/bcell_selection'). Every array fetch is prefixed by it, so the
// same obsm/X_umap + obs/* paths resolve inside whichever group is active.
async function openArray(path: string, group: string, base: string) {
  const rel = group ? `${group}/${path}` : path
  const store = new zarr.FetchStore(`${base}/${rel}`)
  return zarr.open.v3(store, { kind: 'array' })
}

export type Point = { position: [number, number]; index: number }

/** obsm/X_umap -> one Point per cell, for the given group. */
export async function loadUmapCoords(group = ''): Promise<Point[]> {
  const arr = await openArray('obsm/X_umap', group, await coordBase())
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
  const arr = await openArray('obs/barcodes', group, await coordBase())
  const data = (await zarr.get(arr)).data as ArrayLike<string>
  return Array.from(data)
}

export type Categorical = { codes: Int8Array | Int16Array | Int32Array; categories: string[] }

/** An AnnData categorical obs column: per-cell integer `codes` + `categories`. */
export async function loadCategorical(col: Level, group = ''): Promise<Categorical> {
  const base = await coordBase()
  const codesArr = await openArray(`obs/${col}/codes`, group, base)
  const codes = (await zarr.get(codesArr)).data as Int8Array | Int16Array | Int32Array
  const catsArr = await openArray(`obs/${col}/categories`, group, base)
  const categories = Array.from((await zarr.get(catsArr)).data as ArrayLike<string>)
  return { codes, categories }
}

// Views live under the coord (rapids) store; only the root reads genes from the
// vis store (DATA_DIR, the gene-query-optimized layout).
async function geneBase(group: string): Promise<string> {
  return group ? coordBase() : DATA_BASE_URL
}

/** Gene names from the var index column (named per the group's `_index` attr). */
export async function loadGeneNames(group = ''): Promise<string[]> {
  const base = await geneBase(group)
  const rel = group ? `${group}/var` : 'var'
  const varGroup = await zarr.open.v3(new zarr.FetchStore(`${base}/${rel}`), { kind: 'group' })
  const idxCol = (varGroup.attrs['_index'] as string) ?? '_index'
  const arr = await openArray(`var/${idxCol}`, group, base)
  return Array.from((await zarr.get(arr)).data as ArrayLike<string>)
}

export type GeneExpression = { colors: Uint8Array; range: [number, number]; warning?: string }

// A dense one-column read fetches every chunk intersecting the column; refuse
// layouts where that means streaming a large share of X (row-oriented chunks).
const MAX_COLUMN_READ_BYTES = 256 * 1024 * 1024

type GeneSource =
  | { kind: 'dense'; base: string; path: string }
  | { kind: 'csc'; base: string; path: string; nObs: number }

const geneSources = new Map<string, Promise<GeneSource>>()
const cscIndptrs = new Map<string, Promise<number[]>>()

// Opened-array cache: zarr.open fetches the array's zarr.json, so without this
// every gene click pays extra metadata round-trips through the proxy.
type OpenedArray = Awaited<ReturnType<typeof openArray>>
const openedArrays = new Map<string, Promise<OpenedArray>>()

// Caches evict on rejection: a transient failure (backend restart, view still
// uploading) must not poison the entry for the rest of the session.
function openAt(url: string): Promise<OpenedArray> {
  let arr = openedArrays.get(url)
  if (!arr) {
    arr = zarr.open.v3(new zarr.FetchStore(url), { kind: 'array' })
    arr.catch(() => openedArrays.delete(url))
    openedArrays.set(url, arr)
  }
  return arr
}

async function probe(base: string, path: string) {
  const res = await fetch(`${base}/${path}/zarr.json`)
  if (!res.ok) return null
  const meta = await res.json()
  return { node: meta.node_type as string, attrs: (meta.attributes ?? {}) as Record<string, unknown> }
}

// Gene reads resolve to layers/gexp first (the zarrsmith setup), then X itself;
// dense arrays and CSC groups are readable, CSR is not (a column read would scan it all).
async function resolveGeneSource(group: string): Promise<GeneSource> {
  const base = await geneBase(group)
  const prefix = group ? `${group}/` : ''
  for (const path of [`${prefix}layers/gexp`, `${prefix}X`]) {
    const p = await probe(base, path)
    if (!p) continue
    if (p.node === 'array') return { kind: 'dense', base, path }
    if (p.attrs['encoding-type'] === 'csc_matrix') {
      const [nObs] = p.attrs['shape'] as [number, number]
      return { kind: 'csc', base, path, nObs }
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
    src.catch(() => geneSources.delete(group))
    geneSources.set(group, src)
  }
  return src
}

// CSC column = two tiny range reads: data/indices[indptr[j]:indptr[j+1]], scattered
// into a dense per-cell vector. indptr is read once per source and cached.
async function readCscColumn(src: { base: string; path: string; nObs: number }, geneIdx: number): Promise<Float32Array> {
  const url = `${src.base}/${src.path}`
  let indptr = cscIndptrs.get(url)
  if (!indptr) {
    indptr = openAt(`${url}/indptr`)
      .then(arr => zarr.get(arr))
      .then(r => Array.from(r.data as ArrayLike<number | bigint>, Number))
    indptr.catch(() => cscIndptrs.delete(url))
    cscIndptrs.set(url, indptr)
  }
  const ip = await indptr
  const out = new Float32Array(src.nObs)
  const [start, end] = [ip[geneIdx], ip[geneIdx + 1]]
  if (start === end) return out
  const sel = [zarr.slice(start, end)]
  const [vals, rows] = await Promise.all([
    openAt(`${url}/data`).then(a => zarr.get(a, sel)),
    openAt(`${url}/indices`).then(a => zarr.get(a, sel)),
  ])
  const v = vals.data as ArrayLike<number>
  const r = rows.data as ArrayLike<number | bigint>
  for (let i = 0; i < v.length; i++) out[Number(r[i])] = Number(v[i])
  return out
}

async function readDenseColumn(src: { base: string; path: string }, geneIdx: number): Promise<ArrayLike<number>> {
  const arr = await openAt(`${src.base}/${src.path}`)
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
    : await readDenseColumn(src, geneIdx)
  // ramp spans only the expressing cells (v > 0); zeros stay dark so the range
  // isn't wasted on the non-expressing majority
  const n = vals.length
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < n; i++) {
    const v = Number(vals[i]) // int64 stores yield BigInt — normalize before math
    if (v <= 0) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const none = lo === Infinity // gene expressed nowhere
  const span = hi - lo || 1
  const colors = new Uint8Array(n * 3)
  for (let i = 0; i < n; i++) {
    const v = Number(vals[i])
    const [r, g, b] = v > 0 ? exprColor((v - lo) / span) : EXPR_NONE
    colors[i * 3] = r
    colors[i * 3 + 1] = g
    colors[i * 3 + 2] = b
  }
  const range: [number, number] = none ? [0, 0] : [lo, hi]
  // integer dtype = raw counts (dtype comes from the cached open — no data scan)
  const stored = await openAt(src.kind === 'csc' ? `${src.base}/${src.path}/data` : `${src.base}/${src.path}`)
  const warning = String(stored.dtype).includes('int')
    ? 'raw counts (integer dtype): colors follow skewed counts — use a normalized store/layer'
    : undefined
  return { colors, range, warning }
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
    // no-store: a cached listing would resurrect just-deleted views
    const res = await fetch(`${await coordBase()}/groups.json`, { cache: 'no-store' })
    if (!res.ok) return DEFAULT_GROUPS
    const gs = (await res.json()) as Group[]
    return Array.isArray(gs) && gs.length > 0 ? gs : DEFAULT_GROUPS
  } catch {
    return DEFAULT_GROUPS
  }
}
