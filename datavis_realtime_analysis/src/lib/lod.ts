/**
 * Level of detail for the scatter: cells are drawn in a fixed random order, and a level
 * is a prefix of that order — a uniform subsample, so density reads true at every level.
 * Levels halve from n; the drawn level keeps roughly VISIBLE_BUDGET points on screen.
 *
 * `scale` is the zoom relative to the fitted view (2^(zoom - fitZoom)). Dot size keeps
 * total ink constant: radius = INK · scale / √drawn, which grows as you zoom in and
 * shrinks each time a denser level comes in. Opacity falls off once more than DENSE
 * points are on screen, so overlapping millions still read as density.
 */

const VISIBLE_BUDGET = 600_000
const MIN_DRAWN = 20_000
const INK = 500
const MIN_RADIUS_PX = 0.4
const MAX_RADIUS_PX = 8
const DENSE = 300_000
const MIN_OPACITY = 0.5

export function drawOrder(n: number, seed = 4242): Uint32Array {
  const order = new Uint32Array(n)
  for (let i = 0; i < n; i++) order[i] = i
  let s = seed >>> 0
  for (let i = n - 1; i > 0; i--) {
    s = (s + 0x6d2b79f5) >>> 0
    let t = Math.imul(s ^ (s >>> 15), s | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    const j = Math.floor((((t ^ (t >>> 14)) >>> 0) / 4294967296) * (i + 1))
    const tmp = order[i]
    order[i] = order[j]
    order[j] = tmp
  }
  return order
}

export function lodCount(n: number, scale: number): number {
  const want = Math.max(MIN_DRAWN, VISIBLE_BUDGET * scale * scale)
  let m = n
  while (m / 2 >= want) m = Math.ceil(m / 2)
  return m
}

export function dotRadiusPx(drawn: number, scale: number): number {
  return Math.min(MAX_RADIUS_PX, Math.max(MIN_RADIUS_PX, (INK * scale) / Math.sqrt(drawn || 1)))
}

export function dotOpacity(drawn: number, scale: number): number {
  const visible = drawn / Math.max(1, scale * scale)
  return Math.min(1, Math.max(MIN_OPACITY, Math.sqrt(DENSE / visible)))
}
