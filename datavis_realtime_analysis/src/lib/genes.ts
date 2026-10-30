import * as zarr from 'zarrita'
import { cached } from './async'
import { EXPR_NONE, exprColor } from './palette'
import { loadVarNames, openCached, probe, viewRootRows } from './zarrData'

export type GeneExpression = { colors: Uint8Array; range: [number, number]; warning?: string }

type GeneSource = { kind: 'dense'; path: string } | { kind: 'csc'; path: string; nObs: number }

const MAX_COLUMN_READ_BYTES = 256 * 1024 * 1024

const memo = new Map<string, Promise<unknown>>()
const once = <T>(key: string, make: () => Promise<T>) => cached(memo, key, make) as Promise<T>

async function resolveGeneSource(): Promise<GeneSource> {
  for (const path of ['layers/gexp', 'X']) {
    const p = await probe(path)
    if (!p) continue
    if (p.node === 'array') return { kind: 'dense', path }
    if (p.attrs['encoding-type'] === 'csc_matrix') return { kind: 'csc', path, nObs: (p.attrs['shape'] as number[])[0] }
  }
  throw new Error('no gene-readable matrix (need layers/gexp or a dense/CSC X) — run `annizarr add-expr` on the store')
}

const geneSource = () => once('source', resolveGeneSource)

/** Root var names, only when the store has a gene-readable matrix (rejects otherwise). */
export const loadGeneNames = () =>
  once('names', async () => {
    await geneSource()
    return loadVarNames()
  })

async function readCscColumn(src: Extract<GeneSource, { kind: 'csc' }>, geneIdx: number): Promise<Float32Array> {
  const ip = await once(`${src.path}/indptr`, async () =>
    Array.from((await zarr.get(await openCached(`${src.path}/indptr`))).data as ArrayLike<number | bigint>, Number))
  const out = new Float32Array(src.nObs)
  const [start, end] = [ip[geneIdx], ip[geneIdx + 1]]
  if (start === end) return out
  const sel = [zarr.slice(start, end)]
  const [vals, rows] = await Promise.all([
    openCached(`${src.path}/data`).then(a => zarr.get(a, sel)),
    openCached(`${src.path}/indices`).then(a => zarr.get(a, sel)),
  ])
  const v = vals.data as ArrayLike<number>
  const r = rows.data as ArrayLike<number | bigint>
  for (let i = 0; i < v.length; i++) out[Number(r[i])] = Number(v[i])
  return out
}

async function readDenseColumn(path: string, geneIdx: number): Promise<ArrayLike<number>> {
  const arr = await openCached(path)
  const readBytes = arr.shape[0] * arr.chunks[1] * 4
  if (readBytes > MAX_COLUMN_READ_BYTES) {
    throw new Error(`store not laid out for gene queries: one gene read would fetch ~${Math.round(readBytes / 1e6)} MB`)
  }
  return (await zarr.get(arr, [null, geneIdx])).data as ArrayLike<number>
}

async function gatherForView(vals: ArrayLike<number>, group: string): Promise<ArrayLike<number>> {
  if (!group) return vals
  const rows = await viewRootRows(group).catch(() => {
    throw new Error('this view predates gene highlighting (no obs/root_row) — re-run the selection')
  })
  const out = new Float32Array(rows.length)
  for (let i = 0; i < rows.length; i++) out[i] = Number(vals[rows[i]])
  return out
}

function rampColors(vals: ArrayLike<number>): { colors: Uint8Array; range: [number, number] } {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < vals.length; i++) {
    const v = Number(vals[i])
    if (v <= 0) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const span = hi - lo || 1
  const colors = new Uint8Array(vals.length * 3)
  for (let i = 0; i < vals.length; i++) {
    const v = Number(vals[i])
    colors.set(v > 0 ? exprColor((v - lo) / span) : EXPR_NONE, i * 3)
  }
  return { colors, range: lo === Infinity ? [0, 0] : [lo, hi] }
}

/** One gene as per-cell ramp colors over the group's cells; zeros stay dark. */
export async function loadGeneExpression(geneIdx: number, group = ''): Promise<GeneExpression> {
  const src = await geneSource()
  const root = src.kind === 'csc' ? await readCscColumn(src, geneIdx) : await readDenseColumn(src.path, geneIdx)
  const { colors, range } = rampColors(await gatherForView(root, group))
  const stored = await openCached(src.kind === 'csc' ? `${src.path}/data` : src.path)
  const warning = String(stored.dtype).includes('int')
    ? 'raw counts (integer dtype): colors follow skewed counts — use a normalized store/layer'
    : undefined
  return { colors, range, warning }
}
