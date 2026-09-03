import type { CSSProperties } from 'react'
import { exprColor, type RGB } from './zarrData'

export const rgb = (c: RGB) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`

// Theme tokens matched to the spatial-data-explorer (MUI dark): #121212 paper,
// 12%-white dividers, radius 8, blue #1976d2 primary, Allen font via body.
export const DIVIDER = 'rgba(255, 255, 255, 0.12)'
export const PRIMARY = '#1976d2'

export const panel: CSSProperties = {
  background: 'rgba(18, 18, 18, 0.92)',
  border: `1px solid ${DIVIDER}`,
  borderRadius: 8,
  padding: '10px 12px',
  color: '#ddd',
}

export const control: CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid rgba(255, 255, 255, 0.18)',
  borderRadius: 6,
  padding: '4px 6px',
  fontSize: 13,
}

// Filled action button (GPU run, Save to store) — MUI contained-primary look.
export const primaryBtn: CSSProperties = {
  ...{ borderRadius: 6, padding: '6px 10px', fontSize: 13 },
  background: PRIMARY,
  border: `1px solid ${PRIMARY}`,
  color: '#fff',
  fontWeight: 600,
  cursor: 'pointer',
  textAlign: 'left',
}

export const label: CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: '#888',
  marginBottom: 4,
}

export const exprGradient = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
  .map(t => rgb(exprColor(t)))
  .join(', ')})`
