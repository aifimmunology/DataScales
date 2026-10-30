import { useEffect, useState, type DependencyList } from 'react'
import { errorText } from '../lib/async'

export type Loaded<T> = { data: T | null; error: string | null; loading: boolean }

/** Runs `load` whenever `deps` change; a superseded load never lands. `keep` holds the
 * previous data while reloading instead of clearing it. A null `load` clears. */
export function useLoad<T>(load: (() => Promise<T>) | null, deps: DependencyList, keep = false): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({ data: null, error: null, loading: !!load })
  useEffect(() => {
    if (!load) {
      setState({ data: null, error: null, loading: false })
      return
    }
    let stale = false
    setState(s => ({ data: keep ? s.data : null, error: null, loading: true }))
    load().then(
      data => !stale && setState({ data, error: null, loading: false }),
      err => !stale && setState(s => ({ data: keep ? s.data : null, error: errorText(err), loading: false })),
    )
    return () => {
      stale = true
    }
  }, deps)
  return state
}
