import { useState, type PointerEvent } from 'react'

/** A width the user drags from a handle; returns the width and the handle's pointerdown. */
export function useDragWidth(initial: number, min: number, max: number) {
  const [width, setWidth] = useState(initial)
  const startDrag = (e: PointerEvent<HTMLElement>) => {
    const el = e.currentTarget
    const [x0, w0] = [e.clientX, width]
    const move = (ev: globalThis.PointerEvent) => setWidth(Math.min(max, Math.max(min, w0 + ev.clientX - x0)))
    const end = () => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('lostpointercapture', end)
    }
    el.setPointerCapture(e.pointerId)
    el.addEventListener('pointermove', move)
    el.addEventListener('lostpointercapture', end)
    e.preventDefault()
  }
  return [width, startDrag] as const
}
