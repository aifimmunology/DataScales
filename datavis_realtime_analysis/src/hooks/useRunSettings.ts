import { useEffect, useState } from 'react'
import { loadSettings, saveSettings, type EmbeddingInfo, type RunSettings } from '../lib/runSettings'
import { getConfig } from '../lib/zarrData'

const RETRIES = 5
const RETRY_MS = 1000

export function useRunSettings() {
  const [embedding, setEmbedding] = useState<EmbeddingInfo | null>(null)
  const [settings, setSettings] = useState<RunSettings>(() => loadSettings(null))
  useEffect(() => {
    let live = true
    const load = (attempt: number) =>
      getConfig().then(c => {
        if (!live) return
        if (c.embedding) setEmbedding(c.embedding)
        else if (attempt < RETRIES) setTimeout(() => load(attempt + 1), RETRY_MS * 2 ** attempt)
      })
    load(0)
    return () => {
      live = false
    }
  }, [])
  const update = (s: RunSettings) => {
    setSettings(s)
    saveSettings(s)
  }
  return { embedding, settings, update }
}
