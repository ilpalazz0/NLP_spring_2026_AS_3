import './AnalyzeButton.css'

function AnalyzeButton({ onClick, loading, disabled }) {
  return (
    <button
      className="analyze-btn"
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? 'Analyzing...' : 'Analyze →'}
    </button>
  )
}

export default AnalyzeButton
