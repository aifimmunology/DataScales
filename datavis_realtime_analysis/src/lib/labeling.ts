import { isViewNative } from './columns'
import { gatherCategorical, loadCategorical, loadObsColumns } from './zarrData'

/** Working labelset: per-cell codes into `cats` (-1 = unlabeled), per-category counts over
 * the loaded cells, and counts of each category's root cells outside them (views).
 * `cats` keeps the baseline order plus first-seen appends — the backend's save order. */
export type LabelCodes = { cats: string[]; counts: number[]; outside: number[]; codes: Int32Array }

export const MAX_LABEL_LEN = 80

export const validLabelsetName = (n: string) => /^[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}$/.test(n) && n !== 'zarr.json'

function countCodes(codes: ArrayLike<number>, nCats: number): number[] {
  const counts = new Array(nCats).fill(0)
  for (let i = 0; i < codes.length; i++) if (codes[i] >= 0) counts[Number(codes[i])]++
  return counts
}

/** The column as it stands in the store, seen from `group` (n cells); an unknown name starts empty. */
export async function loadWorkingLabels(name: string, group: string, n: number): Promise<LabelCodes> {
  if (isViewNative(group, name)) {
    const own = await loadCategorical(name, group)
    const codes = Int32Array.from(own.codes)
    return { cats: own.categories, counts: countCodes(codes, own.categories.length), outside: [], codes }
  }
  if (!(await loadObsColumns('')).includes(name)) {
    return { cats: [], counts: [], outside: [], codes: new Int32Array(n).fill(-1) }
  }
  const root = await loadCategorical(name, '').catch(() => {
    throw new Error(`'${name}' is an obs column but not a categorical one`)
  })
  const codes = Int32Array.from(group ? (await gatherCategorical(root, group)).codes : root.codes)
  const counts = countCodes(codes, root.categories.length)
  const outside = countCodes(root.codes, root.categories.length).map((c, k) => c - counts[k])
  return { cats: root.categories, counts, outside, codes }
}

export function applyLabel(l: LabelCodes, label: string, indices: number[]): LabelCodes {
  const cats = l.cats.includes(label) ? l.cats : [...l.cats, label]
  const k = cats.indexOf(label)
  const pad = (xs: number[]) => [...xs, ...new Array(cats.length - xs.length).fill(0)]
  const counts = pad(l.counts)
  const codes = Int32Array.from(l.codes)
  for (const i of indices) {
    if (codes[i] >= 0) counts[codes[i]]--
    codes[i] = k
    counts[k]++
  }
  return { cats, counts, outside: pad(l.outside), codes }
}

/** Without categories that have no cells anywhere, codes renumbered — the backend's save. */
export function dropEmpty(l: LabelCodes): LabelCodes {
  const live = l.cats.map((_, k) => (l.counts[k] ?? 0) + (l.outside[k] ?? 0) > 0)
  if (live.every(Boolean)) return l
  const remap = new Int32Array(l.cats.length).fill(-1)
  const kept: number[] = []
  live.forEach((ok, k) => {
    if (ok) remap[k] = kept.push(k) - 1
  })
  const codes = l.codes.map(c => (c >= 0 ? remap[c] : -1))
  const pick = (xs: number[]) => kept.map(k => xs[k] ?? 0)
  return { cats: kept.map(k => l.cats[k]), counts: pick(l.counts), outside: pick(l.outside), codes }
}
