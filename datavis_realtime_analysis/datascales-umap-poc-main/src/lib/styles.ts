import type { CSSProperties } from 'react'
import { exprColor, type RGB } from './zarrData'

export const rgb = (c: RGB) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`

export const panel: CSSProperties = {
  background: 'rgba(20, 20, 20, 0.85)',
  border: '1px solid #333',
  borderRadius: 6,
  padding: '10px 12px',
  color: '#ddd',
  fontFamily: 'system-ui, sans-serif',
}

export const control: CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 4,
  padding: '4px 6px',
  fontSize: 13,
}

export const label: CSSProperties = {
  display: 'block',
  fontSize: 11,
  color: '#888',
  marginBottom: 4,
}

export const exprGradient = `linear-gradient(to right, ${rgb(exprColor(0))}, ${rgb(exprColor(1))})`
