import { useMemo, useState } from 'react'
import { withRetry } from '../lib/async'
import { dropdownLabelSets, groupColumns, isViewNative, pickLevel } from '../lib/columns'
import { gatherCategorical, loadCategorical, loadLabelSets, type Categorical } from '../lib/zarrData'
import { useLoad } from './useLoad'

async function loadColumnSets(group: string) {
  const [own, root] = await Promise.all([loadLabelSets(group), group ? loadLabelSets('') : null])
  return { group, own, root: root ?? own }
}

/** The group's categorical columns (a view's: the root's plus its own leiden) and the Color-by
 * level: the last one picked while the group offers it, else a default. */
export function useColumns(group: string, version: number) {
  const sets = useLoad(() => withRetry(() => loadColumnSets(group), 3), [group, version], true)
  const data = sets.data?.group === group ? sets.data : null
  const columns = useMemo(() => (data ? groupColumns(group, data.own, data.root) : []), [group, data])
  const levels = useMemo(() => dropdownLabelSets(columns), [columns])
  const [wanted, setLevel] = useState('')
  const level = data ? pickLevel(wanted, levels) : ''
  return { columns, rootSets: data?.root ?? [], levels, level, setLevel }
}

function loadLevel(level: string, group: string): Promise<Categorical> {
  if (!group || isViewNative(group, level)) return loadCategorical(level, group)
  return loadCategorical(level, '')
    .then(root => gatherCategorical(root, group))
    .catch(() => loadCategorical(level, group))
}

/** The level's codes for the group's cells: root columns are read live through obs/root_row. */
export const useCategorical = (level: string, group: string, version: number) =>
  useLoad(level ? () => withRetry(() => loadLevel(level, group), 1) : null, [level, group, version])
