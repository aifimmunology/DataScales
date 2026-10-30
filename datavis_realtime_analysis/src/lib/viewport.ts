import { HEADER_HEIGHT } from './styles'

export type Fit = { target: [number, number, number]; zoom: number }

const FALLBACK: Fit = { target: [0, 0, 0], zoom: 3 }

/** Camera fitting interleaved x,y coords into the canvas (window minus header and rail). */
export function fitView(coords: Float32Array): Fit {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (let i = 0; i < coords.length; i += 2) {
    const x = coords[i], y = coords[i + 1]
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  if (!Number.isFinite(minX)) return FALLBACK
  const vw = (window.innerWidth || 800) - HEADER_HEIGHT
  const vh = (window.innerHeight || 600) - HEADER_HEIGHT
  const zoom = Math.log2(Math.min(vw / Math.max(maxX - minX, 1e-6), vh / Math.max(maxY - minY, 1e-6)) * 0.9)
  return { target: [(minX + maxX) / 2, (minY + maxY) / 2, 0], zoom: Math.max(-2, Math.min(zoom, 10)) }
}

export function permutePairs(coords: Float32Array, order: Uint32Array): Float32Array {
  const out = new Float32Array(order.length * 2)
  for (let j = 0; j < order.length; j++) {
    out[j * 2] = coords[order[j] * 2]
    out[j * 2 + 1] = coords[order[j] * 2 + 1]
  }
  return out
}
