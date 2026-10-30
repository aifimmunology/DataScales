import { drawOrder } from '../lib/lod'
import { loadBarcodes, loadUmapCoords } from '../lib/zarrData'
import { useLoad } from './useLoad'

export type Embedding = { coords: Float32Array; barcodes: string[]; order: Uint32Array; n: number }

async function loadEmbedding(group: string): Promise<Embedding> {
  const [coords, barcodes] = await Promise.all([loadUmapCoords(group), loadBarcodes(group)])
  const n = coords.length / 2
  return { coords, barcodes, order: drawOrder(n), n }
}

export const useEmbedding = (group: string, tick: number) => useLoad(() => loadEmbedding(group), [group, tick])
