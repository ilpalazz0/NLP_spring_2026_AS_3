import { useState, useCallback } from 'react'
import { API_URL } from '../../constants'
import './GloVePage.css'

const DATASETS = [
  { key: 'literature', label: 'Literature', desc: 'Trained on Azerbaijani literary corpus' },
  { key: 'imdb',       label: 'IMDB',       desc: 'Trained on IMDB movie review dataset'  },
]


const SAMPLES = [
  { a: 'ata',    b: 'kişi',    c: 'qadın',    label: 'ata − kişi + qadın'       },
  { a: 'qardaş',   b: 'kişi',    c: 'qadın',    label: 'qardaş − kişi + qadın'      },
  { a: 'kitab',  b: 'kitablar',  c: 'evlər', label: 'kitab − kitablar + evlər' },
  { a: 'böyük',  b: 'kiçik',   c: 'uzun',     label: 'böyük − kiçik + uzun'     },
]
function GloVePage() {
  const [dataset,  setDataset]  = useState('literature')
  const [wordA,    setWordA]    = useState('')
  const [wordB,    setWordB]    = useState('')
  const [wordC,    setWordC]    = useState('')
  const [results,  setResults]  = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [equation, setEquation] = useState('')

  const topResult = results?.[0]

  const handleSolve = useCallback(async () => {
    if (!wordA.trim() || !wordB.trim() || !wordC.trim()) return
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const res = await fetch(`${API_URL}/vector_operation`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embedding: 'glove',
          dataset,
          word_a: wordA.trim(),
          word_b: wordB.trim(),
          word_c: wordC.trim(),
          top_k:  8,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Request failed')
      setResults(data.results)
      setEquation(data.equation)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [dataset, wordA, wordB, wordC])

  const handleKey = (e) => { if (e.key === 'Enter') handleSolve() }

  const maxSim = results ? Math.max(...results.map(r => r.similarity)) : 1

  return (
    <div className="page glove-page">

      {/* ── Header ── */}
      <div className="page-header">
        <div className="page-header__tag">EMBEDDINGS</div>
        <h1 className="page-header__title">GloVe <span>Analysis</span></h1>
        <p className="page-header__sub">Solve semantic vector equations using trained GloVe embeddings</p>
      </div>

      {/* ── Dataset toggle ── */}
      <div className="dataset-toggle">
        {DATASETS.map(d => (
          <button
            key={d.key}
            className={`dataset-btn ${dataset === d.key ? 'dataset-btn--active' : ''}`}
            onClick={() => { setDataset(d.key); setResults(null); setError(null) }}
          >
            <span className="dataset-btn__label">{d.label}</span>
            <span className="dataset-btn__desc">{d.desc}</span>
          </button>
        ))}
      </div>


      {/* ── Sample equations ── */}
      <div className="samples-row">
        <span className="samples-label">TRY:</span>
        {SAMPLES.map(s => (
          <button
            key={s.label}
            className="sample-pill"
            onClick={() => { setWordA(s.a); setWordB(s.b); setWordC(s.c); setResults(null); setError(null) }}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* ── Equation builder ── */}

      <div className="equation-card">
        <div className="equation-label">VECTOR EQUATION</div>

        <div className="equation-row">
          <div className="eq-slot">
            <label className="eq-slot__tag">A</label>
            <input
              className="eq-slot__input"
              placeholder="ata"
              value={wordA}
              onChange={e => setWordA(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <span className="eq-op eq-op--minus">−</span>

          <div className="eq-slot">
            <label className="eq-slot__tag">B</label>
            <input
              className="eq-slot__input"
              placeholder="kişi"
              value={wordB}
              onChange={e => setWordB(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <span className="eq-op eq-op--plus">+</span>

          <div className="eq-slot">
            <label className="eq-slot__tag">C</label>
            <input
              className="eq-slot__input"
              placeholder="qadın"
              value={wordC}
              onChange={e => setWordC(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <span className="eq-op eq-op--eq">=</span>

          <div className="eq-slot eq-slot--result">
            {loading ? (
              <span className="eq-spinner" />
            ) : topResult ? (
              <span className="eq-result-word">{topResult.word}</span>
            ) : (
              <span className="eq-result-placeholder">?</span>
            )}
          </div>
        </div>

        <button
          className={`solve-btn ${loading ? 'solve-btn--loading' : ''}`}
          onClick={handleSolve}
          disabled={loading || !wordA.trim() || !wordB.trim() || !wordC.trim()}
        >
          {loading ? 'Solving…' : 'Solve Equation'}
        </button>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="vec-error">
          <span className="vec-error__icon">⚠</span>
          {error}
        </div>
      )}

      {/* ── Results ── */}
      {results && !error && (
        <div className="results-section">
          <div className="results-header">
            <span className="results-equation">{equation} = ?</span>
            <span className="results-count">{results.length} nearest neighbours</span>
          </div>

          <div className="results-list">
            {results.map((r, i) => (
              <div key={r.word} className={`result-row ${i === 0 ? 'result-row--top' : ''}`}>
                <span className="result-rank">#{i + 1}</span>
                <span className="result-word">{r.word}</span>
                <div className="result-bar-track">
                  <div
                    className="result-bar-fill"
                    style={{ width: `${(r.similarity / maxSim) * 100}%` }}
                  />
                </div>
                <span className="result-sim">{r.similarity.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}

export default GloVePage