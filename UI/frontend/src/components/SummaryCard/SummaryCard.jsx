import './SummaryCard.css'

function SummaryCard({ label, value, variant }) {
  return (
    <div className="summary-card">
      <div className="summary-card__label">{label}</div>
      <div className={`summary-card__value summary-card__value--${variant}`}>
        {value}
      </div>
    </div>
  )
}

export default SummaryCard
