import * as zarr from 'zarrita'

// All zarr reads go through the backend's /api/data proxy (Vite in dev, nginx in Docker).
export const DATA_BASE_URL = window.location.origin + '/api/data'

type AppConfig = { store?: string }
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

// Store recorded in selection exports/submissions.
export async function getStoreId(): Promise<string> {
  const cfg = await getConfig()
  return cfg.store || DATA_BASE_URL
}

// A color-by label set is any categorical obs column; discovered per group.
export type Level = string
export type LabelSetInfo = { name: string; own: boolean; nCats: number } // own = app-created (extendable)

/** Every categorical obs column in the group, with app-ownership and category
 * count — metadata-only (counts come from categories/zarr.json shape). Consumers
 * filter: the Color-by dropdown applies its inclusion rule; the Labels panel
 * searches the full list for extending/forking. */
export async function loadLabelSets(group = ''): Promise<LabelSetInfo[]> {
  const base = DATA_BASE_URL
  const prefix = group ? `${group}/` : ''
  const obsMeta = await probe(base, `${prefix}obs`)
  const cols = (obsMeta?.attrs['column-order'] as string[]) ?? []
  const keep = await Promise.all(
    cols.map(async col => {
      const p = await probe(base, `${prefix}obs/${col}`)
      if (p?.attrs['encoding-type'] !== 'categorical') return null
      const res = await fetch(`${base}/${prefix}obs/${col}/categories/zarr.json`)
      if (!res.ok) return null
      const meta = await res.json()
      return {
        name: col,
        own: p.attrs['datavis-labelset'] === true,
        nCats: (meta.shape?.[0] as number) ?? 0,
      }
    }),
  )
  return keep.filter((c): c is LabelSetInfo => c !== null)
}

/** Color-by dropdown rule: leiden + AIFI_L* + app labelsets always, plus any
 * other categorical with < 20 categories. */
export function dropdownLabelSets(sets: LabelSetInfo[]): string[] {
  return sets
    .filter(s => s.own || s.name === 'leiden' || s.name.startsWith('AIFI_L') || s.nCats < 20)
    .map(s => s.name)
}

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
  const arr = await openArray('obsm/X_umap', group, DATA_BASE_URL)
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
  const arr = await openArray('obs/barcodes', group, DATA_BASE_URL)
  const data = (await zarr.get(arr)).data as ArrayLike<string>
  return Array.from(data)
}

export type Categorical = { codes: Int8Array | Int16Array | Int32Array; categories: string[] }

/** An AnnData categorical obs column: per-cell integer `codes` + `categories`. */
export async function loadCategorical(col: Level, group = ''): Promise<Categorical> {
  const base = DATA_BASE_URL
  const codesArr = await openArray(`obs/${col}/codes`, group, base)
  const codes = (await zarr.get(codesArr)).data as Int8Array | Int16Array | Int32Array
  const catsArr = await openArray(`obs/${col}/categories`, group, base)
  const categories = Array.from((await zarr.get(catsArr)).data as ArrayLike<string>)
  return { codes, categories }
}

/** A ROOT categorical gathered into a view's row order via obs/root_row — how app
 * labelsets are read inside views: always the current root state, never the frozen
 * copy the pipeline baked into the view's own obs. */
export async function gatherRootCategorical(col: Level, group: string): Promise<Categorical> {
  const [root, rows] = await Promise.all([loadCategorical(col, ''), viewRootRows(group)])
  const codes = new Int16Array(rows.length)
  for (let i = 0; i < rows.length; i++) codes[i] = Number(root.codes[Number(rows[i])])
  return { codes, categories: root.categories }
}

// Gene data always lives in the ROOT store: GPU views hold only obs + coords, so
// their expression comes from the root gene source gathered via obs/root_row.
let geneNamesP: Promise<string[]> | null = null

/** Gene names from the root var index column (named per its `_index` attr). */
export function loadGeneNames(): Promise<string[]> {
  geneNamesP ??= (async () => {
    const varGroup = await zarr.open.v3(new zarr.FetchStore(`${DATA_BASE_URL}/var`), { kind: 'group' })
    const idxCol = (varGroup.attrs['_index'] as string) ?? '_index'
    const arr = await openArray(`var/${idxCol}`, '', DATA_BASE_URL)
    return Array.from((await zarr.get(arr)).data as ArrayLike<string>)
  })()
  geneNamesP.catch(() => {
    geneNamesP = null // transient failure: retry on next call
  })
  return geneNamesP
}

export type GeneExpression = { colors: Uint8Array; range: [number, number]; warning?: string }

// A dense one-column read fetches every chunk intersecting the column; refuse
// layouts where that means streaming a large share of X (row-oriented chunks).
const MAX_COLUMN_READ_BYTES = 256 * 1024 * 1024

type GeneSource =
  | { kind: 'dense'; base: string; path: string }
  | { kind: 'csc'; base: string; path: string; nObs: number }

let geneSourceP: Promise<GeneSource> | null = null
const cscIndptrs = new Map<string, Promise<number[]>>()
const rootRowMaps = new Map<string, Promise<Int32Array>>() // per view: view row -> root row

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

// Gene reads resolve to the root layers/gexp first (the zarrsmith setup), then
// root X; dense arrays and CSC groups are readable, CSR is not (a column read
// would scan it all).
async function resolveGeneSource(): Promise<GeneSource> {
  for (const path of ['layers/gexp', 'X']) {
    const p = await probe(DATA_BASE_URL, path)
    if (!p) continue
    if (p.node === 'array') return { kind: 'dense', base: DATA_BASE_URL, path }
    if (p.attrs['encoding-type'] === 'csc_matrix') {
      const [nObs] = p.attrs['shape'] as [number, number]
      return { kind: 'csc', base: DATA_BASE_URL, path, nObs }
    }
  }
  throw new Error(
    'no gene-readable matrix (need layers/gexp or a dense/CSC X) — run `zarrsmith add-expr --format csc` on the store',
  )
}

function geneSource(): Promise<GeneSource> {
  geneSourceP ??= resolveGeneSource()
  geneSourceP.catch(() => {
    geneSourceP = null
  })
  return geneSourceP
}

// A view's obs/root_row maps its rows onto the root store (written by the GPU
// pipeline), so root gene columns can be gathered into view order.
function viewRootRows(group: string): Promise<Int32Array> {
  let p = rootRowMaps.get(group)
  if (!p) {
    p = (async () => {
      const arr = await openArray('obs/root_row', group, DATA_BASE_URL)
      return (await zarr.get(arr)).data as Int32Array
    })()
    p.catch(() => rootRowMaps.delete(group))
    rootRowMaps.set(group, p)
  }
  return p
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

/** One gene column (root layers/gexp, dense X, or CSC X) -> per-cell ramp colors
 * + value range. In a view, root values are gathered via obs/root_row, and the
 * ramp normalizes over the view's own cells. */
export async function loadGeneExpression(geneIdx: number, group = ''): Promise<GeneExpression> {
  const src = await geneSource()
  const rootVals = src.kind === 'csc'
    ? await readCscColumn(src, geneIdx)
    : await readDenseColumn(src, geneIdx)
  let vals: ArrayLike<number> = rootVals
  if (group) {
    const rows = await viewRootRows(group).catch(() => {
      throw new Error('this view predates gene highlighting (no obs/root_row) — re-run the selection')
    })
    const gathered = new Float32Array(rows.length)
    for (let i = 0; i < rows.length; i++) gathered[i] = Number(rootVals[Number(rows[i])])
    vals = gathered
  }
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
    const res = await fetch(`${DATA_BASE_URL}/groups.json`, { cache: 'no-store' })
    if (!res.ok) return DEFAULT_GROUPS
    const gs = (await res.json()) as Group[]
    return Array.isArray(gs) && gs.length > 0 ? gs : DEFAULT_GROUPS
  } catch {
    return DEFAULT_GROUPS
  }
}
