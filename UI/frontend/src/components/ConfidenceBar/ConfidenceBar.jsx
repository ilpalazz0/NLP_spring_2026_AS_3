import './ConfidenceBar.css'

function ConfidenceBar({ confidence, label, probability }) {
  return (
    <div className="confidence-bar">
      <div className="confidence-bar__row">
        <span className="confidence-bar__label">Confidence</span>
        <span className="confidence-bar__value">
          {(confidence * 100).toFixed(1)}%
        </span>
      </div>
      <div className="confidence-bar__track">
        <div
          className={`confidence-bar__fill confidence-bar__fill--${label}`}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
      <div className="confidence-bar__prob">
        P(positive) = <span>{probability.toFixed(4)}</span>
      </div>
    </div>
  )
}

export default ConfidenceBar
