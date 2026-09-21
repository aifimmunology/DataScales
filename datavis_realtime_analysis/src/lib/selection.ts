import type { Point } from './zarrData'

// The artifact the RAPIDS side consumes. See data/selection.example.json.
//   store       - the datavis-app store the selection was drawn against (provenance)
//   group       - the active view/embedding path the indices refer to ('' = root)
//   lasso_world - the polygon the user drew, in UMAP coords (the shape)
//   indices     - row numbers of the cells INSIDE the shape
//   barcodes    - those same cells, by obs_names id
export type SelectionArtifact = {
  store: string
  group?: string
  lasso_world: [number, number][]
  indices: number[]
  barcodes: string[]
}

// Ray-casting point-in-polygon, in world (UMAP) coords.
export function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    const hits = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
    if (hits) inside = !inside
  }
  return inside
}

// Cells whose UMAP coordinate falls inside the drawn polygon. Ascending order.
export function selectIndices(points: Point[], poly: [number, number][]): number[] {
  if (poly.length < 3) return []
  const out: number[] = []
  for (let i = 0; i < points.length; i++) {
    const [x, y] = points[i].position
    if (pointInPolygon(x, y, poly)) out.push(points[i].index)
  }
  return out
}

// Trigger a browser download of selection.json.
export function downloadSelection(artifact: SelectionArtifact): void {
  const blob = new Blob([JSON.stringify(artifact, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'selection.json'
  a.click()
  URL.revokeObjectURL(url)
}
