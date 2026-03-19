import './SummaryBar.css'
import SummaryCard from '../SummaryCard/SummaryCard'

function SummaryBar({ results }) {
  const posCount = results.filter(r => r.label === 'positive').length
  const negCount = results.filter(r => r.label === 'negative').length
  const avgConf  = (
    results.reduce((s, r) => s + r.confidence, 0) / results.length * 100
  ).toFixed(1)

  const verdict        = posCount > negCount ? 'POSITIVE'
                       : negCount > posCount ? 'NEGATIVE' : 'TIED'
  const verdictVariant = posCount > negCount ? 'positive'
                       : negCount > posCount ? 'negative' : 'neutral'

  return (
    <div className="summary-bar">
      <SummaryCard label="Verdict"        value={verdict}                          variant={verdictVariant} />
      <SummaryCard label="Positive"       value={`${posCount}/${results.length}`}  variant="positive" />
      <SummaryCard label="Negative"       value={`${negCount}/${results.length}`}  variant="negative" />
      <SummaryCard label="Avg Confidence" value={`${avgConf}%`}                   variant="neutral" />
    </div>
  )
}

export default SummaryBar
