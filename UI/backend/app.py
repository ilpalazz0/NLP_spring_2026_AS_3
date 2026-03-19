DATASET_STATS_CACHE      = None
LITERATURE_STATS_CACHE   = None
EVALUATION_CACHE         = None

import sys
import os

TASK5_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Code/task5'))
sys.path.insert(0, TASK5_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re
import pandas as pd

from data_cleaning import load_and_clean_data, clean_text
from models.RNN import RNNModel
from models.BiRNN import BiRNNModel
from models.LSTM import LSTMModel
from features.count_vectorizer import load_count_vectorizer
from features.tfidf import load_tfidf_vectorizer
from features.pmi import load_pmi_features, text_to_pmi_sequence
from features.word2vec import load_word2vec_embeddings_pkl, text_to_word2vec_sequence
from features.glove import load_glove_embeddings_pkl, text_to_glove_sequence

app = Flask(__name__)
CORS(app)

SAVED_MODELS_DIR  = os.path.join(TASK5_DIR, 'saved_models/')
DATA_PATH         = os.path.join(TASK5_DIR, '../../data/Sentiment/dataset.csv')
LITERATURE_DIR    = os.path.join(TASK5_DIR, '../../data/Literature')
SAMPLE_SIZE      = 10000
MAX_SEQ_LEN      = 100
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Server] Using device: {device}')

MODEL_CONFIGS = [
    ('rnn',   'count',    32, 1, 1),
    ('rnn',   'tfidf',    32, 1, 1),
    ('rnn',   'pmi',      32, 1, 1),
    ('rnn',   'word2vec', 128, 1, 300),
    ('rnn',   'glove',    128, 1, 300),
    ('birnn', 'count',    32, 1, 1),
    ('birnn', 'tfidf',    32, 1, 1),
    ('birnn', 'pmi',      32, 1, 1),
    ('birnn', 'word2vec', 128, 1, 300),
    ('birnn', 'glove',    128, 1, 300),
    ('lstm',  'count',    32, 1, 1),
    ('lstm',  'tfidf',    32, 1, 1),
    ('lstm',  'pmi',      32, 1, 1),
    ('lstm',  'word2vec', 128, 1, 300),
    ('lstm',  'glove',    128, 1, 300),
]

# ==========================
# Vector TXT file paths (same directory as app.py)
# ==========================
_HERE = os.path.dirname(os.path.abspath(__file__))

VECTOR_TXT_PATHS = {
    'word2vec_literature': os.path.join(_HERE, 'word2vec_literature.txt'),
    'word2vec_imdb':       os.path.join(_HERE, 'word2vec_imdb.txt'),
    'glove_literature':    os.path.join(_HERE, 'glove_literature.txt'),
    'glove_imdb':          os.path.join(_HERE, 'glove_imdb.txt'),
}


# ==========================
# Load feature transformers
# ==========================
print('[Server] Loading feature transformers...')
transformers = {}
transformers['count']    = load_count_vectorizer(os.path.join(SAVED_MODELS_DIR, 'count_vectorizer.pkl'))
transformers['tfidf']    = load_tfidf_vectorizer(os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.pkl'))
transformers['pmi']      = load_pmi_features(os.path.join(SAVED_MODELS_DIR, 'pmi_scores.pkl'))
transformers['word2vec'] = load_word2vec_embeddings_pkl(os.path.join(SAVED_MODELS_DIR, 'word2vec_embeddings.pkl'))
transformers['glove']    = load_glove_embeddings_pkl(os.path.join(SAVED_MODELS_DIR, 'glove_embeddings.pkl'))

# Pre-compute OOV fallback vectors once (mean of all embedding vectors)
oov_word2vec = np.mean(list(transformers['word2vec'].values()), axis=0).astype(np.float32)
oov_glove    = np.mean(list(transformers['glove'].values()),    axis=0).astype(np.float32)
print('[Server] Feature transformers loaded')


# ==========================
# Load TXT vector files
# ==========================
def load_vectors_txt(filepath):
    """Load word vectors from a .txt file (word2vec / GloVe text format)."""
    vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            # Skip header line present in some word2vec txt exports (e.g. "30000 300")
            if i == 0 and len(parts) == 2:
                try:
                    int(parts[0]); int(parts[1])
                    continue  # This is a header — skip it
                except ValueError:
                    pass
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors[word] = vector
            except ValueError:
                continue
    return vectors


print('[Server] Loading vector TXT files...')
txt_vectors = {}
for key, path in VECTOR_TXT_PATHS.items():
    if os.path.exists(path):
        txt_vectors[key] = load_vectors_txt(path)
        print(f'  [OK] {key} — {len(txt_vectors[key])} words')
    else:
        print(f'  [SKIP] {key} — file not found: {path}')
print(f'[Server] {len(txt_vectors)} vector sets loaded')


# ==========================
# Load models
# ==========================
print('[Server] Loading all models...')
models = {}
for model_type, feature_type, hidden_size, num_layers, input_size in MODEL_CONFIGS:
    model_file = os.path.join(SAVED_MODELS_DIR, f'{model_type}_{feature_type}.pth')
    if not os.path.exists(model_file):
        print(f'  [SKIP] {model_type}_{feature_type} — file not found')
        continue
    if model_type == 'rnn':
        model = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == 'birnn':
        model = BiRNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == 'lstm':
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    models[f'{model_type}_{feature_type}'] = (model, feature_type)
    print(f'  [OK] {model_type}_{feature_type}')
print(f'[Server] {len(models)} models loaded ✓')


# ==========================
# Helpers
# ==========================
def extract_features(cleaned_text, feature_type):
    """Convert a single cleaned text string into a model-ready tensor."""
    if feature_type == 'count':
        vec = transformers['count'].transform([cleaned_text])
        return torch.tensor(vec.toarray(), dtype=torch.float32).unsqueeze(2).to(device)

    elif feature_type == 'tfidf':
        vec = transformers['tfidf'].transform([cleaned_text])
        return torch.tensor(vec.toarray(), dtype=torch.float32).unsqueeze(2).to(device)

    elif feature_type == 'pmi':
        seq = text_to_pmi_sequence(cleaned_text, transformers['pmi'])
        return pad_sequence(
            [torch.tensor(seq, dtype=torch.float32)],
            batch_first=True
        ).unsqueeze(2).to(device)

    elif feature_type == 'word2vec':
        seq = text_to_word2vec_sequence(cleaned_text, transformers['word2vec'], oov_word2vec)
        return pad_sequence(
            [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)],
            batch_first=True
        ).to(device)

    elif feature_type == 'glove':
        seq = text_to_glove_sequence(cleaned_text, transformers['glove'], oov_glove)
        return pad_sequence(
            [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)],
            batch_first=True
        ).to(device)


def tokenize_azerbaijani(text):
    """Simple Azerbaijani tokenizer."""
    return re.findall(r"[a-zəğıöüşç]+", text.lower())


def cosine_similarity_np(v1, v2):
    """Cosine similarity between two numpy vectors."""
    dot  = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


def solve_vector_equation(word_a, word_b, word_c, vectors, top_k=10):
    """
    Solve: word_a − word_b + word_c = ?
    Returns top_k closest words sorted by cosine similarity.
    """
    missing = [w for w in [word_a, word_b, word_c] if w not in vectors]
    if missing:
        return None, missing

    result  = vectors[word_a] - vectors[word_b] + vectors[word_c]
    exclude = {word_a, word_b, word_c}

    # Vectorised batch cosine similarity for speed
    words      = [w for w in vectors if w not in exclude]
    matrix     = np.stack([vectors[w] for w in words])          # (V, D)
    result_norm = result / (np.linalg.norm(result) + 1e-10)
    norms      = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed     = matrix / (norms + 1e-10)
    sims       = normed @ result_norm                            # (V,)

    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    results = [{'word': words[i], 'similarity': round(float(sims[i]), 4)} for i in top_idx]
    return results, []


def compute_dataset_stats():
    df    = load_and_clean_data(DATA_PATH, sample_size=None)
    total = len(df)
    pos   = int((df['sentiment'] == 1).sum())
    neg   = int((df['sentiment'] == 0).sum())

    all_words = []
    lengths   = []
    for review in df['clean_review']:
        words = tokenize_azerbaijani(review)
        all_words.extend(words)
        lengths.append(len(words))

    counter   = Counter(all_words)
    vocab     = list(counter.keys())
    top_words = [{'word': w, 'count': c} for w, c in counter.most_common(20)]

    # Term-document matrix (top 10 terms × first 5 reviews)
    top_terms      = [w for w, _ in counter.most_common(10)]
    sample_reviews = df['clean_review'].iloc[:5].tolist()
    tdm_matrix     = []
    for term in top_terms:
        row = [tokenize_azerbaijani(doc).count(term) for doc in sample_reviews]
        tdm_matrix.append(row)
    doc_labels = [f'Review {i+1}' for i in range(len(sample_reviews))]

    # Word-word co-occurrence matrix (top 10 terms, all docs)
    wwm = [[0] * 10 for _ in range(10)]
    for review in df['clean_review']:
        words_in_review = set(tokenize_azerbaijani(review))
        for i, w1 in enumerate(top_terms):
            for j, w2 in enumerate(top_terms):
                if w1 in words_in_review and w2 in words_in_review:
                    wwm[i][j] += 1

    return {
        'total':        total,
        'positive':     pos,
        'negative':     neg,
        'positive_pct': round(pos / total * 100, 1),
        'negative_pct': round(neg / total * 100, 1),
        'vocab_size':   len(vocab),
        'avg_length':   round(sum(lengths) / len(lengths), 1),
        'max_length':   max(lengths),
        'top_words':    top_words,
        'term_doc_matrix':  {'terms': top_terms, 'docs': doc_labels, 'matrix': tdm_matrix},
        'word_word_matrix': {'words': top_terms, 'matrix': wwm},
    }


def compute_literature_stats():
    # Load all .md files from author subdirectories
    records = []
    for author in os.listdir(LITERATURE_DIR):
        author_path = os.path.join(LITERATURE_DIR, author)
        if not os.path.isdir(author_path):
            continue
        for file in os.listdir(author_path):
            if file.endswith('.md'):
                file_path = os.path.join(author_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                records.append({'author': author, 'filename': file, 'text': text})

    df      = pd.DataFrame(records)
    total   = len(df)
    authors = sorted(df['author'].unique().tolist())

    # Per-author doc counts
    author_counts = (
        df.groupby('author').size()
        .sort_values(ascending=False)
        .reset_index(name='count')
        .rename(columns={'index': 'author'})
    )
    author_dist = [{'author': row['author'], 'count': int(row['count'])}
                   for _, row in author_counts.iterrows()]

    # Tokenise & compute lengths
    all_words = []
    lengths   = []
    for text in df['text']:
        words = tokenize_azerbaijani(text)
        all_words.extend(words)
        lengths.append(len(words))

    counter   = Counter(all_words)
    vocab     = list(counter.keys())
    top_words = [{'word': w, 'count': c} for w, c in counter.most_common(20)]

    # Term-document matrix — top 10 authors by doc count (aggregated)
    top_terms    = [w for w, _ in counter.most_common(10)]
    top_authors  = [row['author'] for _, row in author_counts.head(10).iterrows()]
    tdm_matrix   = []
    for term in top_terms:
        row = []
        for author in top_authors:
            author_texts = df[df['author'] == author]['text'].tolist()
            count = sum(tokenize_azerbaijani(t).count(term) for t in author_texts)
            row.append(count)
        tdm_matrix.append(row)
    sample_labels = top_authors

    # Word-word co-occurrence matrix — all docs
    wwm = [[0] * 10 for _ in range(10)]
    for text in df['text']:
        words_in_doc = set(tokenize_azerbaijani(text))
        for i, w1 in enumerate(top_terms):
            for j, w2 in enumerate(top_terms):
                if w1 in words_in_doc and w2 in words_in_doc:
                    wwm[i][j] += 1

    return {
        'total':       total,
        'num_authors': len(authors),
        'authors':     authors,
        'author_dist': author_dist,
        'vocab_size':  len(vocab),
        'avg_length':  round(sum(lengths) / len(lengths), 1) if lengths else 0,
        'max_length':  max(lengths) if lengths else 0,
        'top_words':   top_words,
        'term_doc_matrix':  {'terms': top_terms, 'docs': sample_labels, 'matrix': tdm_matrix},
        'word_word_matrix': {'words': top_terms, 'matrix': wwm},
    }


def compute_evaluation():

    df     = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
    split  = int(0.8 * len(df))
    X_test = df['clean_review'].iloc[split:].reset_index(drop=True)
    y_true = df['sentiment'].iloc[split:].values

    count_tensor = torch.tensor(
        transformers['count'].transform(X_test).toarray(),
        dtype=torch.float32
    ).unsqueeze(2).to(device)

    tfidf_tensor = torch.tensor(
        transformers['tfidf'].transform(X_test).toarray(),
        dtype=torch.float32
    ).unsqueeze(2).to(device)

    pmi_tensor = pad_sequence(
        [torch.tensor(text_to_pmi_sequence(t, transformers['pmi']), dtype=torch.float32)
         for t in X_test],
        batch_first=True
    ).unsqueeze(2).to(device)

    w2v_tensor = pad_sequence(
        [torch.tensor(
            text_to_word2vec_sequence(t, transformers['word2vec'], oov_word2vec)[:MAX_SEQ_LEN],
            dtype=torch.float32)
         for t in X_test],
        batch_first=True
    ).to(device)

    glove_tensor = pad_sequence(
        [torch.tensor(
            text_to_glove_sequence(t, transformers['glove'], oov_glove)[:MAX_SEQ_LEN],
            dtype=torch.float32)
         for t in X_test],
        batch_first=True
    ).to(device)

    feature_tensors = {
        'count':    count_tensor,
        'tfidf':    tfidf_tensor,
        'pmi':      pmi_tensor,
        'word2vec': w2v_tensor,
        'glove':    glove_tensor,
    }

    results    = []
    EVAL_BATCH = 64

    for key, (model, feature_type) in models.items():
        X_tensor = feature_tensors[feature_type]
        preds    = []
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], EVAL_BATCH):
                batch = X_tensor[i:i + EVAL_BATCH]
                out   = model(batch)
                preds.append((out >= 0.5).float().cpu())

        preds = torch.cat(preds).numpy()
        acc   = accuracy_score(y_true, preds)
        f1    = f1_score(y_true, preds, zero_division=0)

        model_type, feat = key.split('_', 1)
        results.append({
            'model':    model_type.upper(),
            'feature':  feat,
            'accuracy': round(float(acc), 4),
            'f1':       round(float(f1),  4),
        })

    return {'results': results}


# ==========================
# Routes
# ==========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    results = []

    for key, (model, feature_type) in models.items():
        try:
            X_tensor = extract_features(cleaned, feature_type)
            with torch.no_grad():
                prob       = model(X_tensor).item()
                label      = 'positive' if prob >= 0.5 else 'negative'
                confidence = prob if prob >= 0.5 else 1 - prob

            model_type, feat = key.split('_', 1)
            results.append({
                'model':       model_type.upper(),
                'feature':     feat,
                'label':       label,
                'probability': round(prob, 4),
                'confidence':  round(min(confidence, 0.9999), 4),
            })
        except Exception as e:
            print(f'  [ERROR] {key}: {e}')
            model_type, feat = key.split('_', 1)
            results.append({
                'model':       model_type.upper(),
                'feature':     feat,
                'label':       'error',
                'probability': 0.5,
                'confidence':  0.0,
            })

    return jsonify({'results': results, 'cleaned_text': cleaned})


@app.route('/vector_operation', methods=['POST'])
def vector_operation():
    """
    Solve: word_a − word_b + word_c = ?
    Body JSON:
      { "embedding": "word2vec"|"glove",
        "dataset":   "literature"|"imdb",
        "word_a":    str,
        "word_b":    str,
        "word_c":    str,
        "top_k":     int  (optional, default 10) }
    """
    data      = request.get_json()
    embedding = data.get('embedding', '').strip()   # 'word2vec' or 'glove'
    dataset   = data.get('dataset',   '').strip()   # 'literature' or 'imdb'
    word_a    = data.get('word_a',    '').strip().lower()
    word_b    = data.get('word_b',    '').strip().lower()
    word_c    = data.get('word_c',    '').strip().lower()
    top_k     = int(data.get('top_k', 10))

    if not all([embedding, dataset, word_a, word_b, word_c]):
        return jsonify({'error': 'Missing required fields'}), 400

    key     = f'{embedding}_{dataset}'
    vectors = txt_vectors.get(key)

    if vectors is None:
        return jsonify({'error': f'Vector set "{key}" not loaded — check server file paths'}), 404

    results, missing = solve_vector_equation(word_a, word_b, word_c, vectors, top_k)

    if missing:
        return jsonify({'error': f'Words not in vocabulary: {", ".join(missing)}'}), 422

    return jsonify({
        'equation': f'{word_a} − {word_b} + {word_c}',
        'results':  results,
        'vocab_size': len(vectors),
    })


@app.route('/vector_vocab', methods=['GET'])
def vector_vocab():
    """
    Return the vocabulary list for a given embedding + dataset combo.
    Query params: ?embedding=word2vec&dataset=literature
    Used for frontend autocomplete.
    """
    embedding = request.args.get('embedding', '').strip()
    dataset   = request.args.get('dataset',   '').strip()
    key       = f'{embedding}_{dataset}'
    vectors   = txt_vectors.get(key)

    if vectors is None:
        return jsonify({'error': f'Vector set "{key}" not loaded'}), 404

    return jsonify({'vocab': list(vectors.keys()), 'size': len(vectors)})


@app.route('/evaluation', methods=['GET'])
def evaluation():
    return jsonify(EVALUATION_CACHE)


@app.route('/dataset_stats', methods=['GET'])
def dataset_stats():
    source = request.args.get('source', 'sentiment')
    if source == 'literature':
        return jsonify(LITERATURE_STATS_CACHE)
    return jsonify(DATASET_STATS_CACHE)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':        'ok',
        'device':        str(device),
        'models_loaded': len(models),
        'vector_sets':   list(txt_vectors.keys()),
    })


if __name__ == '__main__':
    print('\n[Server] Starting Flask server on http://localhost:5000')

    print('[Server] Precomputing dataset stats...')
    DATASET_STATS_CACHE = compute_dataset_stats()
    print('[Server] Dataset stats ready')

    print('[Server] Precomputing literature stats...')
    LITERATURE_STATS_CACHE = compute_literature_stats()
    print('[Server] Literature stats ready')

    print('[Server] Precomputing evaluation metrics...')
    EVALUATION_CACHE = compute_evaluation()
    print('[Server] Evaluation ready')

    app.run(debug=False, port=5000)