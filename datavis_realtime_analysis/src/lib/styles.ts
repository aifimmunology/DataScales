import type { CSSProperties } from 'react'
import { exprColor, type RGB } from './palette'

export const rgb = (c: RGB) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`

export const HEADER_HEIGHT = 55
export const DIVIDER = 'rgba(255, 255, 255, 0.12)'
const PRIMARY = '#1976d2'

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

export const button: CSSProperties = { ...control, padding: '6px 10px', cursor: 'pointer', textAlign: 'left' }
export const buttonOff: CSSProperties = { ...button, opacity: 0.4, cursor: 'not-allowed' }
export const dangerButton: CSSProperties = { ...button, color: '#f88', borderColor: '#622' }

export const primaryBtn: CSSProperties = {
  borderRadius: 6,
  padding: '6px 10px',
  fontSize: 13,
  background: PRIMARY,
  border: `1px solid ${PRIMARY}`,
  color: '#fff',
  fontWeight: 600,
  cursor: 'pointer',
  textAlign: 'left',
}

export const ghostBtn: CSSProperties = {
  alignSelf: 'flex-start',
  background: 'none',
  border: `1px solid ${DIVIDER}`,
  color: '#aaa',
  borderRadius: 6,
  padding: '2px 8px',
  fontSize: 11,
  cursor: 'pointer',
}

export const label: CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: '#888',
  marginBottom: 4,
}

export const muted: CSSProperties = { fontSize: 12, color: '#888' }

export const column = (gap: number): CSSProperties => ({ ...panel, display: 'flex', flexDirection: 'column', gap })

export const exprGradient = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
  .map(t => rgb(exprColor(t)))
  .join(', ')})`
