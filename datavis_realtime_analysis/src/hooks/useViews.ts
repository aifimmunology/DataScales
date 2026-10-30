import { useEffect, useState } from 'react'
import { deleteView, fetchViews, type Group } from '../lib/api'
import { errorText, withRetry } from '../lib/async'

export function useViews(notify: (msg: string) => void) {
  const [groups, setGroups] = useState<Group[]>([])
  const [group, setGroup] = useState('')
  const [deleting, setDeleting] = useState(false)

  const refresh = () =>
    fetchViews()
      .then(gs => (setGroups(gs), gs))
      .catch(err => (console.error('Views refresh failed:', err), null))

  useEffect(() => {
    let stale = false
    withRetry(fetchViews, 3)
      .then(gs => !stale && setGroups(gs))
      .catch(err => console.error('Failed to load views:', err))
    return () => {
      stale = true
    }
  }, [])

  const recheck = async (path: string) => {
    const gs = await refresh()
    if (gs && !gs.some(g => g.path === path)) {
      setGroup('')
      notify('That view no longer exists — returned to the full store.')
    }
  }

  const remove = async (id: string) => {
    if (!window.confirm('Delete this view from the store? This cannot be undone.')) return
    setDeleting(true)
    setGroup('')
    try {
      await deleteView(id)
    } catch (err) {
      notify(`Delete failed: ${errorText(err)}`)
    } finally {
      await refresh()
      setDeleting(false)
    }
  }

  return { groups, group, setGroup, deleting, refresh, recheck, remove }
}
