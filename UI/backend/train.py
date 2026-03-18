"""
Run this once before starting the Flask server.
Trains all models and saves them + stats to ./models/
"""

import os, re, pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs('./models', exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('./data/dataset.csv')
df = df.dropna(subset=['review', 'sentiment'])
df = df[df['review'].astype(str).str.strip() != '']
df['sentiment'] = df['sentiment'].astype(int)

texts  = df['review'].astype(str).tolist()
labels = df['sentiment'].tolist()

le = LabelEncoder()
y  = le.fit_transform(labels)
num_classes = len(le.classes_)

X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=42, stratify=y
)

# ── BOW Features ──────────────────────────────────────────────────────────────
MAX_FEATURES = 500

cv            = CountVectorizer(max_features=MAX_FEATURES)
X_train_cv    = cv.fit_transform(X_train_texts).toarray()
X_test_cv     = cv.transform(X_test_texts).toarray()

tfidf         = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = tfidf.fit_transform(X_train_texts).toarray()
X_test_tfidf  = tfidf.transform(X_test_texts).toarray()

def compute_pmi_fit(X_train):
    X         = X_train.astype(np.float32)
    total     = X.sum()
    word_freq = X.sum(axis=0) / total
    return word_freq, total

def compute_pmi_transform(X, word_freq, total):
    X        = X.astype(np.float32)
    doc_freq = X.sum(axis=1, keepdims=True) / total
    pmi      = np.log((X / total) / (doc_freq * word_freq + 1e-10) + 1e-10)
    return np.maximum(pmi, 0)

word_freq, total = compute_pmi_fit(X_train_cv)
X_train_pmi      = compute_pmi_transform(X_train_cv, word_freq, total)
X_test_pmi       = compute_pmi_transform(X_test_cv,  word_freq, total)

# ── Sequence Features ─────────────────────────────────────────────────────────
MAX_VOCAB = 500
MAX_LEN   = 200
PAD_IDX   = 0
UNK_IDX   = 1

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

counter = Counter()
for text in X_train_texts:
    counter.update(tokenize(text))

vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
for word, _ in counter.most_common(MAX_VOCAB):
    vocab[word] = len(vocab)

def encode(text):
    tokens = tokenize(text)[:MAX_LEN]
    ids    = [vocab.get(t, UNK_IDX) for t in tokens]
    ids   += [PAD_IDX] * (MAX_LEN - len(ids))
    return ids

X_train_seq = np.array([encode(t) for t in X_train_texts])
X_test_seq  = np.array([encode(t) for t in X_test_texts])

# ── Models ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            x = self.embedding(x)          # (batch, seq_len, embed_dim)
        else:
            x = x.unsqueeze(1)             # (batch, 1, input_size)
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
            x = self.embedding(x)          # (batch, seq_len, embed_dim)
        else:
            x = x.unsqueeze(1)             # (batch, 1, input_size)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, model, epochs=20, lr=0.001, batch_size=64, use_long=False):
    TensorType = torch.LongTensor if use_long else torch.FloatTensor
    X_t     = TensorType(X_train).to(device)
    y_t     = torch.LongTensor(y_train).to(device)
    X_val_t = TensorType(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    train_dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss, patience_counter = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model

def evaluate_model(X_test, y_test, model, use_long=False):
    TensorType = torch.LongTensor if use_long else torch.FloatTensor
    X_t = TensorType(X_test).to(device)
    y_t = torch.LongTensor(y_test).to(device)
    model.eval()
    with torch.no_grad():
        preds  = model(X_t).argmax(dim=1).cpu().numpy()
        y_true = y_t.cpu().numpy()
    return (
        round(accuracy_score(y_true, preds) * 100, 2),
        round(f1_score(y_true, preds, average='weighted') * 100, 2)
    )

# ── Run all combinations ──────────────────────────────────────────────────────
EMBED_DIM   = 64
hidden_size = 64
vocab_size  = len(vocab)
input_size  = X_train_cv.shape[1]

features = {
    'CountVectorizer' : (X_train_cv,    X_test_cv,    False),
    'TF-IDF'          : (X_train_tfidf, X_test_tfidf, False),
    'PMI'             : (X_train_pmi,   X_test_pmi,   False),
}

results   = []
all_models = {}   # { "RNN_CountVectorizer": model, ... }

from sklearn.model_selection import train_test_split as tts

for feat_name, (X_tr, X_te, use_long) in features.items():
    print(f"\n{'='*50}\nFeature: {feat_name}\n{'='*50}")

    X_tr_split, X_val_split, y_tr_split, y_val_split = tts(
        X_tr, y_train, test_size=0.1, random_state=42
    )

    use_emb = feat_name == 'Sequence'
    emb_kw  = dict(use_embedding=True, vocab_size=vocab_size, embed_dim=EMBED_DIM)

    model_list = [
        ('RNN',               RNNClassifier(input_size, hidden_size, num_classes, **(emb_kw if use_emb else {}))),
        ('Bidirectional RNN', RNNClassifier(input_size, hidden_size, num_classes, bidirectional=True, **(emb_kw if use_emb else {}))),
        ('LSTM',              LSTMClassifier(input_size, hidden_size, num_classes, **(emb_kw if use_emb else {}))),
    ]

    for model_name, model in model_list:
        print(f"\n  Training {model_name}...")
        trained = train_model(X_tr_split, y_tr_split, X_val_split, y_val_split,
                              model, epochs=20, use_long=use_long)

        acc, f1 = evaluate_model(X_te, y_test, trained, use_long=use_long)
        print(f"  Accuracy: {acc}%  |  F1: {f1}%")

        key = f"{model_name}_{feat_name}"
        all_models[key] = trained
        results.append({'feature': feat_name, 'model': model_name, 'accuracy': acc, 'f1': f1})

# ── Save everything ───────────────────────────────────────────────────────────
torch.save({k: m.state_dict() for k, m in all_models.items()}, './models/weights.pt')

pickle.dump({
    'cv': cv, 'tfidf': tfidf,
    'vocab': vocab, 'word_freq': word_freq, 'total': total,
    'le': le, 'results': results,
    'num_classes': num_classes,
    'input_size': input_size,
    'vocab_size': vocab_size,
}, open('./models/artifacts.pkl', 'wb'))

print("\nAll models saved to ./models/")
print(pd.DataFrame(results).pivot(index='model', columns='feature', values='accuracy').to_string())