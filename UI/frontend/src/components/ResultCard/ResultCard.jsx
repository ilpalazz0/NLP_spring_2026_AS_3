import './ResultCard.css'
import ConfidenceBar from '../ConfidenceBar/ConfidenceBar'

function ResultCard({ result, index }) {
  const { model, feature, label, confidence, probability } = result

  return (
    <div
      className={`result-card result-card--${label}`}
      style={{ animationDelay: `${index * 40}ms` }}
    >
      <div className="result-card__header">
        <div className="result-card__model">{model}</div>
        <div className="result-card__feature">{feature}</div>
      </div>
      <div className={`result-card__sentiment result-card__sentiment--${label}`}>
        {label === 'positive' ? '▲ Positive' : '▼ Negative'}
      </div>
      <ConfidenceBar
        confidence={confidence}
        label={label}
        probability={probability}
      />
    </div>
  )
}

export default ResultCard
