import './EmptyState.css'

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">⬡</div>
      <p className="empty-state__text">
        Enter a review and click Analyze to see predictions from all 15 models
      </p>
    </div>
  )
}

export default EmptyState
