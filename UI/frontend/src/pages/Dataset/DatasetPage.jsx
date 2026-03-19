import { useState, useEffect } from 'react'
import './DatasetPage.css'
import { API_URL } from '../../constants'

const SOURCES = [
  { key: 'sentiment',  label: 'Sentiment',  desc: 'Azerbaijani sentiment dataset' },
  { key: 'literature', label: 'Literature', desc: 'Azerbaijani literary corpus'   },
]

function StatCard({ label, value, sub }) {
  return (
    <div className="stat-card">
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__value">{value}</div>
      {sub && <div className="stat-card__sub">{sub}</div>}
    </div>
  )
}

function SectionTitle({ children }) {
  return <h2 className="section-title">{children}</h2>
}

function TermDocMatrix({ data }) {
  if (!data) return null
  const { terms, docs, matrix } = data
  return (
    <div className="matrix-wrap">
      <div className="matrix-scroll">
        <table className="matrix-table">
          <thead>
            <tr>
              <th className="matrix-table__corner">Term / Doc</th>
              {docs.map((d, i) => (
                <th key={i} className="matrix-table__doc-header">{d}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {terms.map((term, ti) => (
              <tr key={ti}>
                <td className="matrix-table__term">{term}</td>
                {matrix[ti].map((val, di) => (
                  <td
                    key={di}
                    className="matrix-table__cell"
                    style={{ '--intensity': Math.min(val / 5, 1) }}
                  >
                    {val > 0 ? val : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function WordWordMatrix({ data }) {
  if (!data) return null
  const { words, matrix } = data
  const maxVal = Math.max(...matrix.flat())
  return (
    <div className="matrix-wrap">
      <div className="matrix-scroll">
        <table className="matrix-table">
          <thead>
            <tr>
              <th className="matrix-table__corner">Word</th>
              {words.map((w, i) => (
                <th key={i} className="matrix-table__doc-header">{w}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {words.map((word, wi) => (
              <tr key={wi}>
                <td className="matrix-table__term">{word}</td>
                {matrix[wi].map((val, ci) => (
                  <td
                    key={ci}
                    className={`matrix-table__cell ${wi === ci ? 'matrix-table__cell--diag' : ''}`}
                    style={{ '--intensity': maxVal > 0 ? val / maxVal : 0 }}
                  >
                    {val > 0 ? val : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TopWordsChart({ words }) {
  if (!words || words.length === 0) return null
  const max = words[0].count
  return (
    <div className="top-words">
      {words.map((w, i) => (
        <div key={i} className="top-words__row">
          <span className="top-words__rank">{i + 1}</span>
          <span className="top-words__word">{w.word}</span>
          <div className="top-words__bar-track">
            <div
              className="top-words__bar-fill"
              style={{ width: `${(w.count / max) * 100}%` }}
            />
          </div>
          <span className="top-words__count">{w.count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  )
}

function AuthorDistChart({ data }) {
  if (!data || data.length === 0) return null
  const max = data[0].count
  return (
    <div className="top-words">
      {data.map((a, i) => (
        <div key={i} className="top-words__row">
          <span className="top-words__rank">{i + 1}</span>
          <span className="top-words__word top-words__word--author">{a.author}</span>
          <div className="top-words__bar-track">
            <div
              className="top-words__bar-fill top-words__bar-fill--author"
              style={{ width: `${(a.count / max) * 100}%` }}
            />
          </div>
          <span className="top-words__count">{a.count}</span>
        </div>
      ))}
    </div>
  )
}

function DatasetPage() {
  const [source,  setSource]  = useState('sentiment')
  const [stats,   setStats]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    setLoading(true)
    setError('')
    setStats(null)
    fetch(`${API_URL}/dataset_stats?source=${source}`)
      .then(r => r.json())
      .then(d => { setStats(d); setLoading(false) })
      .catch(() => { setError('Could not load dataset stats'); setLoading(false) })
  }, [source])

  const isLit = source === 'literature'

  return (
    <div className="page dataset-page">
      <div className="page-header">
        <div className="page-header__tag">CORPUS ANALYSIS</div>
        <h1 className="page-header__title">Dataset <span>Statistics</span></h1>
        <p className="page-header__sub">Explore and compare the two training corpora</p>
      </div>

      {/* ── Source toggle ── */}
      <div className="source-toggle">
        {SOURCES.map(s => (
          <button
            key={s.key}
            className={`source-btn ${source === s.key ? 'source-btn--active' : ''}`}
            onClick={() => setSource(s.key)}
          >
            <span className="source-btn__label">{s.label}</span>
            <span className="source-btn__desc">{s.desc}</span>
          </button>
        ))}
      </div>

      {loading && <div className="loading-msg">Loading dataset statistics…</div>}
      {error   && <div className="error-msg">{error}</div>}

      {stats && !loading && !error && (
        <>
          {/* ── Stat cards ── */}
          <div className="stat-grid">
            <StatCard label="Total Documents" value={stats.total.toLocaleString()} />
            {isLit ? (
              <StatCard label="Authors" value={stats.num_authors} />
            ) : (
              <>
                <StatCard label="Positive" value={stats.positive.toLocaleString()} sub={`${stats.positive_pct}%`} />
                <StatCard label="Negative" value={stats.negative.toLocaleString()} sub={`${stats.negative_pct}%`} />
              </>
            )}
            <StatCard label="Vocabulary Size"   value={stats.vocab_size.toLocaleString()} />
            <StatCard label="Avg Doc Length"     value={`${stats.avg_length} words`} />
            <StatCard label="Max Doc Length"     value={`${stats.max_length} words`} />
          </div>

          {/* ── Author distribution (literature only) ── */}
          {isLit && stats.author_dist && (
            <>
              <SectionTitle>Documents per Author</SectionTitle>
              <AuthorDistChart data={stats.author_dist} />
            </>
          )}

          {/* ── Top words ── */}
          <SectionTitle>Top 20 Most Frequent Words</SectionTitle>
          <TopWordsChart words={stats.top_words} />

          {/* ── Term-doc matrix ── */}
          <SectionTitle>
            Term-Document Matrix{' '}
            <span className="section-title__sub">(top 10 terms × 5 sample docs)</span>
          </SectionTitle>
          <p className="section-desc">Shows how often each top term appears in a sample of documents.</p>
          <TermDocMatrix data={stats.term_doc_matrix} />

          {/* ── Word-word matrix ── */}
          <SectionTitle>
            Word-Word Co-occurrence Matrix{' '}
            <span className="section-title__sub">(top 10 words)</span>
          </SectionTitle>
          <p className="section-desc">Shows how often pairs of frequent words appear in the same document.</p>
          <WordWordMatrix data={stats.word_word_matrix} />
        </>
      )}
    </div>
  )
}

export default DatasetPage