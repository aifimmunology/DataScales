import { useEffect, useMemo, useState } from 'react'
import { UNLABELED_CODE, UNLABELED_NAME } from '../lib/palette'
import { fromCodes, fromIndices, type Selection } from '../lib/selection'
import type { Categorical } from '../lib/zarrData'

const toggle = <T,>(s: Set<T>, v: T) => {
  const next = new Set(s)
  if (next.has(v)) next.delete(v)
  else next.add(v)
  return next
}

function codesFor(cat: Categorical, names: Set<string>): Set<number> {
  const out = new Set<number>()
  cat.categories.forEach((name, k) => names.has(name) && out.add(k))
  if (names.has(UNLABELED_NAME)) out.add(UNLABELED_CODE)
  return out
}

/** The highlighted cells, from a lasso, legend categories (kept by name across views), or
 * working-label rows. The three sources are mutually exclusive. */
export function useSelection(n: number, group: string, level: string, cat: Categorical | null) {
  const [selection, setSelection] = useState<Selection | null>(null)
  const [cats, setCats] = useState<Set<string>>(new Set())
  const [labelRows, setLabelRows] = useState<Set<number>>(new Set())

  useEffect(() => {
    setSelection(null)
    setLabelRows(new Set())
  }, [group])
  useEffect(() => setCats(new Set()), [level])
  useEffect(() => {
    if (cat && n > 0 && cat.codes.length === n && cats.size > 0) setSelection(fromCodes(cat.codes, codesFor(cat, cats)))
  }, [cat, n])

  const legendCodes = useMemo(() => (cat ? codesFor(cat, cats) : new Set<number>()), [cat, cats])

  const clear = () => {
    setSelection(null)
    setCats(new Set())
    setLabelRows(new Set())
  }
  const lasso = (indices: number[]) => {
    clear()
    setSelection(fromIndices(n, indices))
  }
  const toggleCategory = (code: number) => {
    if (!cat) return
    const next = toggle(cats, code < 0 ? UNLABELED_NAME : cat.categories[code])
    setCats(next)
    setLabelRows(new Set())
    setSelection(fromCodes(cat.codes, codesFor(cat, next)))
  }
  const toggleLabelRow = (codes: Int32Array, k: number) => {
    const next = toggle(labelRows, k)
    setLabelRows(next)
    setCats(new Set())
    setSelection(fromCodes(codes, next))
  }
  const dropLasso = () => {
    if (cats.size === 0 && labelRows.size === 0) setSelection(null)
  }

  return { selection, legendCodes, labelRows, clear, lasso, toggleCategory, toggleLabelRow, dropLasso }
}
