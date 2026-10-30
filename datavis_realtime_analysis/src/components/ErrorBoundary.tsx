import { Component, type ReactNode } from 'react'

export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div style={style}>
        <div>The app hit an error: {this.state.error.message}</div>
        <button onClick={() => window.location.reload()} style={buttonStyle}>
          Reload
        </button>
      </div>
    )
  }
}

const style: React.CSSProperties = {
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  gap: 12,
  alignItems: 'center',
  justifyContent: 'center',
  color: '#f88',
  background: '#111',
  fontSize: 15,
}

const buttonStyle: React.CSSProperties = {
  background: '#1c1c1c',
  color: '#ddd',
  border: '1px solid #444',
  borderRadius: 6,
  padding: '6px 14px',
  cursor: 'pointer',
}
