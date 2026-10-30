export type Selection = { mask: Uint8Array; indices: number[] }

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Cells (rows of interleaved x,y coords) inside a world-space polygon, ascending. */
export function lassoIndices(coords: Float32Array, poly: [number, number][]): number[] {
  if (poly.length < 3) return []
  const out: number[] = []
  for (let i = 0; i < coords.length / 2; i++) {
    if (pointInPolygon(coords[i * 2], coords[i * 2 + 1], poly)) out.push(i)
  }
  return out
}

export function fromIndices(n: number, indices: number[]): Selection {
  const mask = new Uint8Array(n)
  for (const i of indices) mask[i] = 1
  return { mask, indices }
}

export function fromCodes(codes: ArrayLike<number>, wanted: Set<number>): Selection | null {
  if (wanted.size === 0) return null
  const mask = new Uint8Array(codes.length)
  const indices: number[] = []
  for (let i = 0; i < codes.length; i++) {
    if (wanted.has(Number(codes[i]))) {
      mask[i] = 1
      indices.push(i)
    }
  }
  return { mask, indices }
}
