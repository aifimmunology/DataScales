import { useEffect, useState } from 'react'
import { fetchGpuHealth, type GpuHealth } from '../lib/api'

const SLOW_POLL_MS = 600_000
const PROBING_POLL_MS = 3000

export function useGpuHealth() {
  const [health, setHealth] = useState<GpuHealth | null>(null)
  const check = (refresh = false) =>
    fetchGpuHealth(refresh)
      .then(h => h && setHealth(h))
      .catch(() => {})
  useEffect(() => {
    check()
    const t = setInterval(() => check(), SLOW_POLL_MS)
    return () => clearInterval(t)
  }, [])
  useEffect(() => {
    if (!health?.checking) return
    const t = setInterval(() => check(), PROBING_POLL_MS)
    return () => clearInterval(t)
  }, [health?.checking])
  return { health, check, problem: health?.status === 'error' }
}
