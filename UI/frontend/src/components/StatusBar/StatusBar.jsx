import './StatusBar.css'

function StatusBar({ message, loading, isError }) {
  const modifier = loading ? 'status-bar--loading'
                 : isError ? 'status-bar--error'
                 : ''
  return (
    <div className={`status-bar ${modifier}`}>
      {loading && <div className="status-bar__spinner" />}
      {message}
    </div>
  )
}

export default StatusBar
