import { useMemo, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import LassoOverlay from './LassoOverlay'
import { dotOpacity, dotRadiusPx, lodCount } from '../lib/lod'
import { fitView, permutePairs } from '../lib/viewport'

type Props = {
  coords: Float32Array
  order: Uint32Array
  colors: Uint8Array
  dotScale: number
  selecting: boolean
  onLasso: (world: [number, number][]) => void
}

const VIEW = new OrthographicView({ id: 'umap' })

/** The embedding, drawn as a prefix of `order` sized to the zoom (see lib/lod). `colors`
 * is RGBA in draw order. Remount per embedding: the camera fits once, on mount. */
export default function ScatterCanvas({ coords, order, colors, dotScale, selecting, onLasso }: Props) {
  const deckRef = useRef<any>(null)
  const fit = useMemo(() => fitView(coords), [coords])
  const [viewState, setViewState] = useState<any>(fit)
  const positions = useMemo(() => permutePairs(coords, order), [coords, order])

  const scale = 2 ** (viewState.zoom - fit.zoom)
  const drawn = lodCount(order.length, scale)
  const data = useMemo(
    () => ({
      length: drawn,
      attributes: { getPosition: { value: positions, size: 2 }, getFillColor: { value: colors, size: 4 } },
    }),
    [drawn, positions, colors],
  )
  const layer = new ScatterplotLayer({
    id: 'umap-scatter',
    data,
    radiusUnits: 'pixels',
    radiusScale: dotRadiusPx(drawn, scale) * dotScale,
    opacity: dotOpacity(drawn, scale),
    pickable: false,
  })

  const unproject = (screen: [number, number][]) => {
    const vp = deckRef.current?.deck?.getViewports?.()[0]
    return vp ? screen.map(p => vp.unproject(p).slice(0, 2) as [number, number]) : null
  }

  return (
    <>
      <DeckGL
        ref={deckRef}
        views={VIEW}
        viewState={viewState}
        onViewStateChange={e => setViewState(e.viewState)}
        controller={{ dragPan: !selecting }}
        layers={[layer]}
        style={{ width: '100%', height: '100%' }}
      />
      {selecting && (
        <LassoOverlay
          onPolygon={screen => {
            const world = unproject(screen)
            if (world) onLasso(world)
          }}
        />
      )}
    </>
  )
}
