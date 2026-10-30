import * as zarr from 'zarrita'
import { cached } from './async'
import type { LabelSetInfo } from './columns'
import type { EmbeddingInfo } from './runSettings'

const DATA_BASE_URL = window.location.origin + '/api/data'

const REVALIDATE: RequestInit = { cache: 'no-cache' }
const fetchStore = (url: string) => new zarr.FetchStore(url, { overrides: REVALIDATE })
const inGroup = (group: string, path: string) => (group ? `${group}/${path}` : path)

type AppConfig = { store?: string; embedding?: EmbeddingInfo }
let cfgPromise: Promise<AppConfig> | null = null

export function getConfig(): Promise<AppConfig> {
  cfgPromise ??= fetch(window.location.origin + '/api/config')
    .then(res => (res.ok ? res.json() : Promise.reject(new Error(`config ${res.status}`))))
    .catch(() => {
      cfgPromise = null
      return {}
    })
  return cfgPromise
}

/** A node's zarr.json, or null when it does not exist; other failures throw. */
export async function probe(path: string) {
  const res = await fetch(`${DATA_BASE_URL}/${path}/zarr.json`, REVALIDATE)
  if (res.status === 404) return null
  if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`)
  const meta = await res.json()
  return {
    node: meta.node_type as string,
    shape: meta.shape as number[] | undefined,
    attrs: (meta.attributes ?? {}) as Record<string, unknown>,
  }
}

const openArray = (path: string) => zarr.open.v3(fetchStore(`${DATA_BASE_URL}/${path}`), { kind: 'array' })
type OpenedArray = Awaited<ReturnType<typeof openArray>>

const openedArrays = new Map<string, Promise<OpenedArray>>()
const rootRowMaps = new Map<string, Promise<Int32Array>>()

export const openCached = (path: string) => cached(openedArrays, path, () => openArray(path))

const readAll = async (path: string) => (await zarr.get(await openArray(path))).data

/** obsm/X_umap as interleaved float32 x,y. */
export async function loadUmapCoords(group = ''): Promise<Float32Array> {
  const data = (await readAll(inGroup(group, 'obsm/X_umap'))) as Float32Array | Float64Array
  return data instanceof Float32Array ? data : Float32Array.from(data)
}

async function readIndex(group: string, node: 'obs' | 'var'): Promise<string[]> {
  const meta = await probe(inGroup(group, node))
  const col = (meta?.attrs['_index'] as string) ?? '_index'
  return Array.from((await readAll(inGroup(group, `${node}/${col}`))) as ArrayLike<string>)
}

export const loadBarcodes = (group = '') => readIndex(group, 'obs')
export const loadVarNames = () => readIndex('', 'var')

export type Categorical = { codes: Int8Array | Int16Array | Int32Array; categories: string[] }

export async function loadCategorical(col: string, group = ''): Promise<Categorical> {
  const [codes, cats] = await Promise.all([
    readAll(inGroup(group, `obs/${col}/codes`)),
    readAll(inGroup(group, `obs/${col}/categories`)),
  ])
  return { codes: codes as Categorical['codes'], categories: Array.from(cats as ArrayLike<string>) }
}

/** A view's obs/root_row: view row -> root-store row. */
export const viewRootRows = (group: string) =>
  cached(rootRowMaps, group, async () => (await readAll(`${group}/obs/root_row`)) as Int32Array)

export async function gatherCategorical(root: Categorical, group: string): Promise<Categorical> {
  const rows = await viewRootRows(group)
  const codes = new Int32Array(rows.length)
  for (let i = 0; i < rows.length; i++) codes[i] = Number(root.codes[rows[i]])
  return { codes, categories: root.categories }
}

export async function loadObsColumns(group = ''): Promise<string[]> {
  const meta = await probe(inGroup(group, 'obs'))
  return (meta?.attrs['column-order'] as string[]) ?? []
}

/** Every categorical obs column in the group, with app ownership and category count (metadata only). */
export async function loadLabelSets(group = ''): Promise<LabelSetInfo[]> {
  const cols = await loadObsColumns(group)
  const found = await Promise.all(
    cols.map(async name => {
      const col = await probe(inGroup(group, `obs/${name}`))
      if (col?.attrs['encoding-type'] !== 'categorical') return null
      const cats = await probe(inGroup(group, `obs/${name}/categories`))
      return cats ? { name, own: col.attrs['datavis-labelset'] === true, nCats: cats.shape?.[0] ?? 0 } : null
    }),
  )
  return found.filter((c): c is LabelSetInfo => c !== null)
}
