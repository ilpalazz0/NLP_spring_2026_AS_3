import { useState, useEffect } from 'react'
import './EvaluationPage.css'
import { API_URL } from '../../constants'

const MODELS   = ['RNN', 'BIRNN', 'LSTM']
const FEATURES = ['count', 'tfidf', 'pmi', 'word2vec', 'glove']

function EvaluationPage() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')
  const [metric, setMetric]   = useState('accuracy')

  useEffect(() => {
    fetch(`${API_URL}/evaluation`)
      .then(r => r.json())
      .then(d => { setResults(d.results); setLoading(false) })
      .catch(() => { setError('Could not load evaluation — is the Flask server running?'); setLoading(false) })
  }, [])

  const getValue = (model, feature) => {
    if (!results) return null
    const r = results.find(
      x => x.model.toLowerCase() === model.toLowerCase() &&
           x.feature.toLowerCase() === feature.toLowerCase()
    )
    return r ? r[metric] : null
  }

  const getBg = val => {
    if (val === null) return 'transparent'
    if (val >= 0.9)  return 'rgba(0, 166, 81,  0.18)'
    if (val >= 0.7)  return 'rgba(0, 102, 255, 0.12)'
    if (val >= 0.55) return 'rgba(255, 165, 0,  0.10)'
    return 'rgba(232, 22, 58, 0.08)'
  }

  const getColor = val => {
    if (val === null) return 'var(--subtext)'
    if (val >= 0.9)  return 'var(--pos)'
    if (val >= 0.7)  return 'var(--accent)'
    if (val >= 0.55) return '#d97706'
    return 'var(--neg)'
  }

  return (
    <div className="page eval-page">
      <div className="page-header">
        <div className="page-header__tag">PERFORMANCE</div>
        <h1 className="page-header__title">Model <span>Evaluation</span></h1>
        <p className="page-header__sub">Accuracy and F1 scores across all 15 model configurations</p>
      </div>

      <div className="metric-toggle">
        <button
          className={`metric-btn ${metric === 'accuracy' ? 'metric-btn--active' : ''}`}
          onClick={() => setMetric('accuracy')}
        >
          Accuracy
        </button>
        <button
          className={`metric-btn ${metric === 'f1' ? 'metric-btn--active' : ''}`}
          onClick={() => setMetric('f1')}
        >
          F1 Score
        </button>
      </div>

      {loading && <div className="loading-msg">Loading evaluation results...</div>}
      {error   && <div className="error-msg">{error}</div>}

      {results && (
        <>
          <div className="matrix-wrap eval-matrix-wrap">
            <table className="eval-matrix">
              <thead>
                <tr>
                  <th className="eval-matrix__corner">Model \ Feature</th>
                  {FEATURES.map(f => (
                    <th key={f} className="eval-matrix__col-header">{f}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MODELS.map(model => (
                  <tr key={model}>
                    <td className="eval-matrix__row-header">{model}</td>
                    {FEATURES.map(feature => {
                      const val = getValue(model, feature)
                      return (
                        <td
                          key={feature}
                          className="eval-matrix__cell"
                          style={{
                            background: getBg(val),
                            color: getColor(val),
                          }}
                        >
                          {val !== null ? (val * 100).toFixed(1) + '%' : '—'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="eval-legend">
            <span className="legend-item legend-item--great">≥ 90%</span>
            <span className="legend-item legend-item--good">≥ 70%</span>
            <span className="legend-item legend-item--ok">≥ 55%</span>
            <span className="legend-item legend-item--poor">&lt; 55%</span>
          </div>

          <h2 className="section-title" style={{ marginTop: 40 }}>All Results</h2>
          <div className="results-table-wrap">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Feature</th>
                  <th>Accuracy</th>
                  <th>F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {results
                  .slice()
                  .sort((a, b) => b.accuracy - a.accuracy)
                  .map((r, i) => (
                    <tr key={i} className={i === 0 ? 'results-table__best' : ''}>
                      <td>{r.model}</td>
                      <td><span className="feature-tag">{r.feature}</span></td>
                      <td style={{ color: getColor(r.accuracy) }}>{(r.accuracy * 100).toFixed(2)}%</td>
                      <td style={{ color: getColor(r.f1) }}>{(r.f1 * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

export default EvaluationPage
