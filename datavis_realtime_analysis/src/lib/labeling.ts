/** Working labelset: per-cell codes into `cats` (-1 = unlabeled), per-category counts over
 * the loaded cells, and counts of each category's root cells outside them (views).
 * `cats` is never pruned while labeling — the baseline order plus first-seen appends,
 * exactly what the backend builds — so the shown order matches the saved one. */
export type LabelCodes = { cats: string[]; counts: number[]; outside: number[]; codes: Int32Array }

/** Copy without categories that have no cells anywhere, codes renumbered — the backend's save. */
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

export function countCodes(codes: ArrayLike<number>, nCats: number): number[] {
  const counts = new Array(nCats).fill(0)
  for (let i = 0; i < codes.length; i++) if (codes[i] >= 0) counts[Number(codes[i])]++
  return counts
}

/** Same rule as the backend's labelset NAME regex. */
export const validLabelsetName = (n: string) => /^[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}$/.test(n) && n !== 'zarr.json'

export const MAX_LABEL_LEN = 80
