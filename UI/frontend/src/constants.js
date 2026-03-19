export const API_URL     = 'http://localhost:5000'
export const MODEL_ORDER = ['rnn', 'birnn', 'lstm']
export const FEAT_ORDER  = ['count', 'tfidf', 'pmi', 'word2vec', 'glove']

export const FILTERS = [
  { key: 'all',      label: 'ALL',      type: 'model' },
  { key: 'rnn',      label: 'RNN',      type: 'model' },
  { key: 'birnn',    label: 'BiRNN',    type: 'model' },
  { key: 'lstm',     label: 'LSTM',     type: 'model' },
  { key: 'word2vec', label: 'Word2Vec', type: 'feature' },
  { key: 'glove',    label: 'GloVe',    type: 'feature' },
]