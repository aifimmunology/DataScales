export type ComboOption = { value: string; label?: string; hint?: string }

export function pickOnEnter(matches: ComboOption[], q: string, canCreate: boolean): ComboOption | null {
  return matches.find(o => o.value === q) ?? (!canCreate && matches.length === 1 ? matches[0] : null)
}
