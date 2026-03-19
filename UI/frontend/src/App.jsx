import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './App.css'
import Navbar          from './components/Navbar/Navbar'
import AnalyzerPage    from './pages/Analyzer/AnalyzerPage'
import DatasetPage     from './pages/Dataset/DatasetPage'
import EvaluationPage  from './pages/Evaluation/EvaluationPage'
import Word2VecPage    from './pages/Word2Vec/Word2VecPage'
import GloVePage       from './pages/GloVe/GloVePage'

function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <Navbar />
        <main className="layout__content">
          <Routes>
            <Route path="/"           element={<Navigate to="/analyzer" replace />} />
            <Route path="/analyzer"   element={<AnalyzerPage />} />
            <Route path="/dataset"    element={<DatasetPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/word2vec"   element={<Word2VecPage />} />
            <Route path="/glove"      element={<GloVePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
