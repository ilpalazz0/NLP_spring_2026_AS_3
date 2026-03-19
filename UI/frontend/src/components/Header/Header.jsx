import './Header.css'

function Header() {
  return (
    <div className="header">
      <h1 className="header__title">
        Azerbaijani<br /><span>Sentiment</span> Analyzer
      </h1>
      <p className="header__subtitle">
        15 models — RNN · BiRNN · LSTM × Count · TF-IDF · PMI · Word2Vec · GloVe
      </p>
    </div>
  )
}

export default Header
