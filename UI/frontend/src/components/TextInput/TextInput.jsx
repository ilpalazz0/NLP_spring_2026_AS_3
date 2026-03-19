import './TextInput.css'
import AnalyzeButton from '../AnalyzeButton/AnalyzeButton'

const SAMPLES = [
  {
    label: 'Positive #1',
    text: 'bu film həqiqətən əla idi, aktyorların oyunu mükəmməl idi və hekayə çox maraqlı idi',
  },
  {
    label: 'Positive #2',
    text: 'çox gözəl film, bütün ailəmlə birlikdə izlədik və hamımız çox xoşladıq',
  },
  {
    label: 'Negative #1',
    text: 'bu filmi izləmək vaxt itkisi idi, heç bir şey məntiqi deyildi və aktyor oyunu dəhşətli idi',
  },
  {
    label: 'Negative #2',
    text: 'həyatımda izlədiyim ən pis filmlərdən biri, heç kimə tövsiyə etmərəm',
  },
  {
    label: 'Mixed',
    text: 'film bəzi yerlərdə maraqlı idi amma ümumiyyətlə gözlədiyimdən zəif çıxdı',
  },
  {
    label: 'Short',
    text: 'əla',
  },
]

function TextInput({ value, onChange, onSubmit, loading }) {
  const handleKey = e => {
    if (e.ctrlKey && e.key === 'Enter') onSubmit()
  }

  return (
    <div className="text-input">
      <label className="text-input__label">Input Text</label>

      <div className="text-input__samples">
        <span className="text-input__samples-label">Try a sample:</span>
        <div className="text-input__samples-list">
          {SAMPLES.map(s => (
            <button
              key={s.label}
              className={`sample-btn ${value === s.text ? 'sample-btn--active' : ''}`}
              onClick={() => onChange(s.text)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <textarea
        className="text-input__textarea"
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="Azerbaijani review yazın... (e.g. bu film çox yaxşı idi)"
      />
      <div className="text-input__footer">
        <span className="text-input__count">
          {value.length} chars · Ctrl+Enter to analyze
        </span>
        <AnalyzeButton
          onClick={onSubmit}
          loading={loading}
          disabled={!value.trim()}
        />
      </div>
    </div>
  )
}

export default TextInput