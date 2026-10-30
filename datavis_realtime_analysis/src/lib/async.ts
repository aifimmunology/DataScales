export const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err))

export async function withRetry<T>(fn: () => Promise<T>, retries: number, delayMs = 1500): Promise<T> {
  for (let left = retries; ; left--) {
    try {
      return await fn()
    } catch (err) {
      if (left <= 0) throw err
      await new Promise(resolve => setTimeout(resolve, delayMs))
    }
  }
}

/** Memoize a promise by key; a rejected entry is dropped so the next call retries. */
export function cached<T>(cache: Map<string, Promise<T>>, key: string, make: () => Promise<T>): Promise<T> {
  let p = cache.get(key)
  if (!p) {
    p = make()
    p.catch(() => cache.delete(key))
    cache.set(key, p)
  }
  return p
}
