export type RGB = [number, number, number]

const PALETTE: RGB[] = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
]

const UNASSIGNED: RGB = [130, 130, 130]

export function colorForCode(code: number): RGB {
  return code < 0 ? UNASSIGNED : PALETTE[code % PALETTE.length]
}

export const UNLABELED_CODE = -1
export const UNLABELED_NAME = '\u0000unlabeled'
export const UNLABELED_SWATCH: RGB = [0, 0, 0]

export const UNCOLORED: RGB = [130, 70, 255]
export const SELECTED: RGB = [255, 240, 30]
export const EXPR_NONE: RGB = [46, 46, 58]

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
  const [a, b] = [RAMP[i], RAMP[i + 1]]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}
