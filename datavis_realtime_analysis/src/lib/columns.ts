export type LabelSetInfo = { name: string; own: boolean; nCats: number }

const VIEW_NATIVE = 'leiden'
const PREFERRED_LEVEL = 'AIFI_L1'
const MAX_DROPDOWN_CATS = 100

export const isViewNative = (group: string, name: string) => !!group && name === VIEW_NATIVE

export const columnLabel = (group: string, name: string) =>
  isViewNative(group, name) ? `${name} (current view only)` : name

/** Columns as seen from a group: the root's, with a view's own VIEW_NATIVE in place of the root's. */
export function groupColumns(group: string, ls: LabelSetInfo[], rootLs: LabelSetInfo[]): LabelSetInfo[] {
  if (!group) return ls
  const own = ls.find(s => s.name === VIEW_NATIVE)
  const cols = rootLs.flatMap(s => (s.name !== VIEW_NATIVE ? [s] : own ? [own] : []))
  return own && !rootLs.some(s => s.name === VIEW_NATIVE) ? [...cols, own] : cols
}

export const dropdownLabelSets = (sets: LabelSetInfo[]) =>
  sets.filter(s => s.nCats < MAX_DROPDOWN_CATS).map(s => s.name)

export function pickLevel(current: string, names: string[]): string {
  if (names.includes(current)) return current
  return names.includes(PREFERRED_LEVEL) ? PREFERRED_LEVEL : names[0] ?? ''
}
