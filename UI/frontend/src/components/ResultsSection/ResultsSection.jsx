import { useState, useMemo } from 'react'
import SummaryBar  from '../SummaryBar/SummaryBar'
import FilterTabs  from '../FilterTabs/FilterTabs'
import ResultsGrid from '../ResultsGrid/ResultsGrid'
import { FILTERS } from '../../constants'

function ResultsSection({ results }) {
  const [filter, setFilter] = useState('all')

  const filtered = useMemo(() => {
    if (filter === 'all') return results

    const filterDef = FILTERS.find(f => f.key === filter)
    if (!filterDef) return results

    if (filterDef.type === 'model') {
      return results.filter(r => r.model.toLowerCase() === filter)
    }

    if (filterDef.type === 'feature') {
      return results.filter(r => r.feature.toLowerCase() === filter)
    }

    return results
  }, [results, filter])

  return (
    <>
      <SummaryBar results={results} />
      <FilterTabs active={filter} onChange={setFilter} />
      <ResultsGrid results={filtered} />
    </>
  )
}

export default ResultsSection