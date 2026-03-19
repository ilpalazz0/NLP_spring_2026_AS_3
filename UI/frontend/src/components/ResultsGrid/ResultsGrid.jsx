import './ResultsGrid.css'
import ResultCard from '../ResultCard/ResultCard'

function ResultsGrid({ results }) {
  return (
    <div className="results-grid">
      {results.map((r, i) => (
        <ResultCard
          key={`${r.model}_${r.feature}`}
          result={r}
          index={i}
        />
      ))}
    </div>
  )
}

export default ResultsGrid
