import './FilterTabs.css'
import { FILTERS } from '../../constants'

function FilterTabs({ active, onChange }) {
  const modelFilters   = FILTERS.filter(f => f.type === 'model')
  const featureFilters = FILTERS.filter(f => f.type === 'feature')

  return (
    <div className="filter-tabs">
      <div className="filter-tabs__group">
        <span className="filter-tabs__group-label">Model</span>
        <div className="filter-tabs__row">
          {modelFilters.map(f => (
            <button
              key={f.key}
              className={`filter-tab ${active === f.key ? 'filter-tab--active' : ''}`}
              onClick={() => onChange(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="filter-tabs__divider" />

      <div className="filter-tabs__group">
        <span className="filter-tabs__group-label">Feature</span>
        <div className="filter-tabs__row">
          {featureFilters.map(f => (
            <button
              key={f.key}
              className={`filter-tab filter-tab--feature ${active === f.key ? 'filter-tab--active' : ''}`}
              onClick={() => onChange(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default FilterTabs