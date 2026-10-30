import { useEffect, useState } from 'react'
import { loadSettings, saveSettings, type EmbeddingInfo, type RunSettings } from '../lib/runSettings'
import { getConfig } from '../lib/zarrData'

export function useRunSettings() {
  const [embedding, setEmbedding] = useState<EmbeddingInfo | null>(null)
  const [settings, setSettings] = useState<RunSettings>(() => loadSettings(null))
  useEffect(() => {
    getConfig().then(c => c.embedding && setEmbedding(c.embedding))
  }, [])
  const update = (s: RunSettings) => {
    setSettings(s)
    saveSettings(s)
  }
  return { embedding, settings, update }
}
