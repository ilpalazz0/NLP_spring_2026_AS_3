"""
Flask backend — run after train.py has completed.
    pip install flask flask-cors
    python app.py
"""

import re, pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load saved artifacts ──────────────────────────────────────────────────────
arts       = pickle.load(open('./models/artifacts.pkl', 'rb'))
cv         = arts['cv']
tfidf      = arts['tfidf']
vocab      = arts['vocab']
word_freq  = arts['word_freq']
total      = arts['total']
le         = arts['le']
results    = arts['results']
num_classes  = arts['num_classes']
input_size   = arts['input_size']
vocab_size   = arts['vocab_size']

PAD_IDX = 0
UNK_IDX = 1
MAX_LEN = 200
EMBED_DIM   = 64
HIDDEN_SIZE = 64

device = torch.device('cpu')   # always CPU for inference

# ── Model definitions (must match train.py exactly) ───────────────────────────
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
                 bidirectional=False, num_layers=2,
                 use_embedding=False, vocab_size=None, embed_dim=None):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=PAD_IDX)
            rnn_input = embed_dim
        else:
            rnn_input = input_size
        self.rnn = nn.RNN(rnn_input, hidden_size, batch_first=True,
                          bidirectional=bidirectional, num_layers=num_layers, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        else:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2,
                 use_embedding=False, vocab_size=None, embed_dim=None):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=PAD_IDX)
            lstm_input = embed_dim
        else:
            lstm_input = input_size
        self.lstm = nn.LSTM(lstm_input, hidden_size, batch_first=True,
                            num_layers=num_layers, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        else:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ── Rebuild model instances & load weights ────────────────────────────────────
def build_models():
    models = {}
    for feat in ['CountVectorizer', 'TF-IDF', 'PMI']:
        models[f'RNN_{feat}']               = RNNClassifier(input_size, HIDDEN_SIZE, num_classes)
        models[f'Bidirectional RNN_{feat}'] = RNNClassifier(input_size, HIDDEN_SIZE, num_classes, bidirectional=True)
        models[f'LSTM_{feat}']              = LSTMClassifier(input_size, HIDDEN_SIZE, num_classes)
    return models

all_models = build_models()
weights    = torch.load('./models/weights.pt', map_location='cpu')
for key, model in all_models.items():
    model.load_state_dict(weights[key])
    model.eval()

print("✅ All models loaded.")

# ── Feature helpers ───────────────────────────────────────────────────────────
def compute_pmi_transform(X, wf, tot):
    X        = X.astype(np.float32)
    doc_freq = X.sum(axis=1, keepdims=True) / tot
    pmi      = np.log((X / tot) / (doc_freq * wf + 1e-10) + 1e-10)
    return np.maximum(pmi, 0)

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def encode(text):
    tokens = tokenize(text)[:MAX_LEN]
    ids    = [vocab.get(t, UNK_IDX) for t in tokens]
    ids   += [PAD_IDX] * (MAX_LEN - len(ids))
    return ids

def featurize(text):
    cv_vec    = cv.transform([text]).toarray()
    tfidf_vec = tfidf.transform([text]).toarray()
    pmi_vec   = compute_pmi_transform(cv_vec, word_freq, total)
    seq_vec   = np.array([encode(text)])
    return {
        'CountVectorizer': (cv_vec,    False),
        'TF-IDF':          (tfidf_vec, False),
        'PMI':             (pmi_vec,   False),
        'Sequence':        (seq_vec,   True),
    }

def predict_one(model, x_np, use_long):
    TType = torch.LongTensor if use_long else torch.FloatTensor
    x = TType(x_np).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))
    label = le.inverse_transform([pred])[0]
    return {
        'label':      'Positive' if label == 1 else 'Negative',
        'confidence': round(float(max(probs)) * 100, 1),
        'probs':      {str(le.classes_[i]): round(float(p)*100,1) for i,p in enumerate(probs)}
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    """Returns pre-computed accuracy + F1 for all 12 model/feature combos."""
    return jsonify({'results': results})

@app.route('/predict', methods=['POST'])
def predict():
    """Runs all 12 models on the submitted text and returns predictions."""
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    features   = featurize(text)
    predictions = []

    for feat_name, (x_np, use_long) in features.items():
        for model_name in ['RNN', 'Bidirectional RNN', 'LSTM']:
            key    = f'{model_name}_{feat_name}'
            result = predict_one(all_models[key], x_np, use_long)
            predictions.append({
                'feature':    feat_name,
                'model':      model_name,
                'label':      result['label'],
                'confidence': result['confidence'],
                'probs':      result['probs'],
            })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, port=5000)