import { useEffect, useMemo, useRef, useState } from 'react'
import { saveLabels } from '../lib/api'
import { errorText } from '../lib/async'
import { isViewNative } from '../lib/columns'
import { applyLabel, dropEmpty, loadWorkingLabels, MAX_LABEL_LEN, type LabelCodes } from '../lib/labeling'

type Pending = { label: string; indices: number[] }
export type Saved = { name: string; labeled: number }

/** Editing one categorical column in place: load it, assign labels to selections, save.
 * `onStored(name)` fires after every save attempt (null on failure: a mirror error still saved locally). */
export function useLabeling(group: string, barcodes: string[], notify: (msg: string) => void, onStored: (name: string | null) => void) {
  const [working, setWorking] = useState<(LabelCodes & { name: string }) | null>(null)
  const [pending, setPending] = useState<Pending[]>([])
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState<Saved | null>(null)
  const token = useRef(0)
  const groupRef = useRef(group)
  groupRef.current = group

  const reset = () => {
    setWorking(null)
    setPending([])
  }
  useEffect(reset, [group])

  const shown = useMemo(() => (working ? { name: working.name, ...dropEmpty(working) } : null), [working])

  const start = async (name: string) => {
    const t = ++token.current
    const g = group
    setSaved(null)
    try {
      const l = await loadWorkingLabels(name, g, barcodes.length)
      if (t !== token.current || groupRef.current !== g) return
      setWorking({ name, ...l })
      setPending([])
    } catch (err) {
      if (t === token.current) notify(`Could not load '${name}': ${errorText(err)}`)
    }
  }

  const assign = (raw: string, indices: number[]) => {
    if (!working || saving || indices.length === 0) return
    const label = raw.slice(0, MAX_LABEL_LEN)
    setWorking({ name: working.name, ...applyLabel(working, label, indices) })
    setPending(p => [...p, { label, indices }])
  }

  const save = async () => {
    if (!working || pending.length === 0) return
    const g = group
    setSaving(true)
    try {
      const assignments = pending.map(p => ({ label: p.label, barcodes: p.indices.map(i => barcodes[i] ?? '') }))
      const res = await saveLabels(working.name, assignments, isViewNative(g, working.name) ? g : undefined)
      setSaved({ name: res.name, labeled: res.labeled })
      if (groupRef.current === g) {
        reset()
        onStored(res.name)
      }
    } catch (err) {
      notify(`Label save failed: ${errorText(err)}`)
      if (groupRef.current === g) onStored(null)
    } finally {
      setSaving(false)
    }
  }

  return {
    shown,
    dirty: pending.length > 0,
    saving,
    saved,
    dismissSaved: () => setSaved(null),
    start,
    assign,
    save,
    cancel: reset,
  }
}
