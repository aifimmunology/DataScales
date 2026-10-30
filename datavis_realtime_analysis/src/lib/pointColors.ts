import { colorForCode, SELECTED, UNCOLORED, type RGB } from './palette'

type ColorInputs = {
  codes?: ArrayLike<number> | null
  expr?: Uint8Array | null
  labels?: Int32Array | null
  mask?: Uint8Array | null
}

const BASE_ALPHA = 180
const LABELED_ALPHA = 235
const UNLABELED_ALPHA = 55
const SELECTED_ALPHA = 255
const UNSELECTED_ALPHA = 40

/** Per-cell RGBA in draw order. Gene expression wins over codes; a working labelset paints
 * over both and dims the rest; a selection paints over everything. */
export function pointColors(order: Uint32Array, { codes, expr, labels, mask }: ColorInputs): Uint8Array {
  const n = order.length
  const fits = <T extends ArrayLike<number>>(a: T | null | undefined, k = 1) => (a && a.length === n * k ? a : null)
  const ex = fits(expr, 3)
  const cs = ex ? null : fits(codes)
  const ls = fits(labels)
  const ms = fits(mask)
  const out = new Uint8Array(n * 4)
  for (let j = 0; j < n; j++) {
    const i = order[j]
    let c: RGB | readonly number[] = cs ? colorForCode(Number(cs[i])) : UNCOLORED
    let a = BASE_ALPHA
    if (ex) c = [ex[i * 3], ex[i * 3 + 1], ex[i * 3 + 2]]
    if (ls) {
      if (ls[i] >= 0) {
        c = colorForCode(ls[i])
        a = LABELED_ALPHA
      } else a = Math.min(a, UNLABELED_ALPHA)
    }
    if (ms) {
      if (ms[i]) {
        c = SELECTED
        a = SELECTED_ALPHA
      } else a = UNSELECTED_ALPHA
    }
    out[j * 4] = c[0]
    out[j * 4 + 1] = c[1]
    out[j * 4 + 2] = c[2]
    out[j * 4 + 3] = a
  }
  return out
}
