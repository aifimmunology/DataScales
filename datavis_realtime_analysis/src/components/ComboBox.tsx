import { useState } from 'react'
import { control } from '../lib/styles'

export type ComboOption = { value: string; label?: string; hint?: string }

type Props = {
  options: ComboOption[]
  value?: string // shown while closed
  placeholder?: string
  onPick: (value: string) => void
  createLabel?: (query: string) => string | null // extra row for a typed name not in options
  onCreate?: (query: string) => void
}

/** Searchable dropdown: focusing lists every option, typing filters them. */
export default function ComboBox({ options, value = '', placeholder, onPick, createLabel, onCreate }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')

  const q = query.trim()
  const shownValue = options.find(o => o.value === value)?.label ?? value
  const matches = q ? options.filter(o => (o.label ?? o.value).toLowerCase().includes(q.toLowerCase())) : options
  const create = q && onCreate && createLabel ? createLabel(q) : null

  const close = () => {
    setOpen(false)
    setQuery('')
  }
  const choose = (fn: () => void) => {
    fn()
    close()
    ;(document.activeElement as HTMLElement | null)?.blur()
  }

  return (
    <div>
      <input
        value={open ? query : shownValue}
        placeholder={open && value ? shownValue : placeholder}
        onFocus={() => setOpen(true)}
        onBlur={close}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Escape') (e.target as HTMLInputElement).blur()
          if (e.key !== 'Enter') return
          const hit = matches.find(o => o.value === q) ?? (matches.length === 1 ? matches[0] : null)
          if (hit) choose(() => onPick(hit.value))
          else if (create) choose(() => onCreate!(q))
        }}
        style={inputStyle}
      />
      {open && (matches.length > 0 || create) && (
        <div style={listStyle} onMouseDown={e => e.preventDefault() /* keep focus through the click */}>
          {matches.map(o => (
            <div
              key={o.value}
              style={{ ...itemStyle, color: o.value === value ? '#ffe94d' : '#ddd' }}
              onClick={() => choose(() => onPick(o.value))}
            >
              {o.label ?? o.value} {o.hint && <span style={{ color: '#888' }}>{o.hint}</span>}
            </div>
          ))}
          {create && (
            <div style={{ ...itemStyle, color: '#8bf' }} onClick={() => choose(() => onCreate!(q))}>
              {create}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const inputStyle: React.CSSProperties = { ...control, width: '100%', boxSizing: 'border-box', cursor: 'text' }

const listStyle: React.CSSProperties = {
  marginTop: 2,
  maxHeight: 240,
  overflowY: 'auto',
  border: '1px solid #333',
  borderRadius: 4,
  background: '#1c1c1c',
}

const itemStyle: React.CSSProperties = {
  padding: '4px 8px',
  fontSize: 13,
  cursor: 'pointer',
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
}
