import { useEffect, useState } from 'react'
import { loadGeneExpression, loadGeneNames } from '../lib/genes'
import { useLoad } from './useLoad'

export function useGenes(group: string) {
  const names = useLoad(loadGeneNames, [])
  const genes = names.data ?? []
  const [gene, setGene] = useState<string | null>(null)
  useEffect(() => setGene(null), [group])
  const idx = gene ? genes.indexOf(gene) : -1
  const expr = useLoad(idx >= 0 ? () => loadGeneExpression(idx, group) : null, [idx, group])
  return { genes, gene, setGene, expr }
}
