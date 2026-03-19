import { useState } from 'react'
import './AnalyzerPage.css'
import { API_URL, MODEL_ORDER, FEAT_ORDER } from '../../constants'
import Header         from '../../components/Header/Header'
import TextInput      from '../../components/TextInput/TextInput'
import StatusBar      from '../../components/StatusBar/StatusBar'
import ResultsSection from '../../components/ResultsSection/ResultsSection'
import EmptyState     from '../../components/EmptyState/EmptyState'

function AnalyzerPage() {
  const [text, setText]       = useState('')
  const [results, setResults] = useState([])
  const [status, setStatus]   = useState('')
  const [loading, setLoading] = useState(false)
  const [isError, setIsError] = useState(false)

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setIsError(false)
    setStatus('Analyzing across all 15 models...')
    setResults([])

    try {
      const res  = await fetch(`${API_URL}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text }),
      })
      const data = await res.json()

      const sorted = (data.results || []).sort((a, b) => {
        const mi = MODEL_ORDER.indexOf(a.model.toLowerCase())
                 - MODEL_ORDER.indexOf(b.model.toLowerCase())
        if (mi !== 0) return mi
        return FEAT_ORDER.indexOf(a.feature) - FEAT_ORDER.indexOf(b.feature)
      })

      setResults(sorted)
      setStatus(`Analysis complete — ${sorted.length} models responded`)
    } catch {
      setIsError(true)
      setStatus('Connection error — is the Flask server running on port 5000?')
    }

    setLoading(false)
  }

  return (
    <div className="page analyzer-page">
      <Header />
      <TextInput
        value={text}
        onChange={setText}
        onSubmit={analyze}
        loading={loading}
      />
      <StatusBar
        message={status}
        loading={loading}
        isError={isError}
      />
      {results.length > 0
        ? <ResultsSection results={results} />
        : !loading && <EmptyState />
      }
    </div>
  )
}

export default AnalyzerPage
