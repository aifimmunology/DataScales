import ErrorBoundary from './components/ErrorBoundary'
import Umap from './components/Umap'

export default function App() {
  return (
    <div style={{ width: '100%', height: '100%', background: '#111', position: 'relative' }}>
      <ErrorBoundary>
        <Umap />
      </ErrorBoundary>
    </div>
  )
}
