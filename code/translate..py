import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim.downloader as api
import re, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════
# ★  YOUR DATASET CONFIG  (only edit here)
# ══════════════════════════════════════════
CSV_PATH     = "/content/imdb_eng.csv"   # ← change to your actual file path
TEXT_COLUMN  = "review"               # ← your text column
LABEL_COLUMN = "sentiment"            # ← your label column
# Labels are: "positive" / "negative"
# ══════════════════════════════════════════

MAX_LEN    = 200
VOCAB_SIZE = 10000
EMBED_DIM  = 100
EPOCHS     = 5
BATCH_SIZE = 64

# ──────────────────────────────────────────
# 1. LOAD & CLEAN
# ──────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# Encode: positive → 1, negative → 0
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df[LABEL_COLUMN])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(f"Total samples : {len(df)}")
print(df[LABEL_COLUMN].value_counts().to_string())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)           # remove HTML tags like <br />
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

texts = df[TEXT_COLUMN].tolist()
labels = df['label_enc'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\nTrain: {len(X_train_text)} | Test: {len(X_test_text)}")

# ──────────────────────────────────────────
# 2. SHARED TOKENIZER
# ──────────────────────────────────────────
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

def to_padded(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

# ──────────────────────────────────────────
# 3. FEATURE EXTRACTORS
# ──────────────────────────────────────────

# --- Count Vectorizer ---
cv = CountVectorizer(max_features=VOCAB_SIZE)
X_tr_cv = cv.fit_transform(X_train_text).toarray()[:, :MAX_LEN].reshape(-1, MAX_LEN, 1).astype(np.float32)
X_te_cv = cv.transform(X_test_text).toarray()[:, :MAX_LEN].reshape(-1, MAX_LEN, 1).astype(np.float32)

# --- TF-IDF ---
tfidf = TfidfVectorizer(max_features=VOCAB_SIZE)
X_tr_tfidf = tfidf.fit_transform(X_train_text).toarray()[:, :MAX_LEN].reshape(-1, MAX_LEN, 1).astype(np.float32)
X_te_tfidf = tfidf.transform(X_test_text).toarray()[:, :MAX_LEN].reshape(-1, MAX_LEN, 1).astype(np.float32)

# --- PMI ---
def compute_pmi_vectors(train_texts, test_texts):
    cv_pmi = CountVectorizer(max_features=VOCAB_SIZE)
    X_counts = cv_pmi.fit_transform(train_texts)
    cooc = (X_counts.T @ X_counts).toarray().astype(np.float32)
    total = cooc.sum()
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((cooc * total) / (row_sum * col_sum + 1e-9) + 1e-9)
    pmi = np.clip(pmi, 0, None)
    vocab = cv_pmi.vocabulary_
    def encode(texts):
        seqs = pad_sequences(tokenizer.texts_to_sequences(texts),
                             maxlen=MAX_LEN, padding='post', truncating='post')
        out = np.zeros((len(texts), MAX_LEN, 1), dtype=np.float32)
        for i, seq in enumerate(seqs):
            for j, idx in enumerate(seq):
                word = tokenizer.index_word.get(idx)
                if word and word in vocab:
                    out[i, j, 0] = pmi[vocab[word]].mean()
        return out
    return encode(train_texts), encode(test_texts)

X_tr_pmi, X_te_pmi = compute_pmi_vectors(X_train_text, X_test_text)

# --- Word2Vec (pretrained, 300-dim) ---
print("\nLoading Word2Vec (this may take a minute)...")
w2v = api.load("word2vec-google-news-300")

def to_embed(texts, model, dim):
    out = np.zeros((len(texts), MAX_LEN, dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, word in enumerate(text.split()[:MAX_LEN]):
            if word in model:
                out[i, j] = model[word]
    return out

X_tr_w2v = to_embed(X_train_text, w2v, 300)
X_te_w2v = to_embed(X_test_text,  w2v, 300)

# --- GloVe (pretrained, 100-dim) ---
print("Loading GloVe (this may take a minute)...")
glove = api.load("glove-twitter-100")

X_tr_glove = to_embed(X_train_text, glove, EMBED_DIM)
X_te_glove = to_embed(X_test_text,  glove, EMBED_DIM)

# ──────────────────────────────────────────
# 4. MODEL BUILDERS
# ──────────────────────────────────────────
def build_rnn(input_shape):
    m = Sequential([
        SimpleRNN(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_birnn(input_shape):
    m = Sequential([
        Bidirectional(SimpleRNN(64), input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_lstm(input_shape):
    m = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ──────────────────────────────────────────
# 5. TRAIN & EVALUATE ALL 15 COMBINATIONS
# ──────────────────────────────────────────
feature_sets = {
    "Count Vectorizer": (X_tr_cv,    X_te_cv),
    "TF-IDF":           (X_tr_tfidf, X_te_tfidf),
    "PMI":              (X_tr_pmi,   X_te_pmi),
    "Word2Vec":         (X_tr_w2v,   X_te_w2v),
    "GloVe":            (X_tr_glove, X_te_glove),
}
model_builders = {
    "RNN":               build_rnn,
    "Bidirectional RNN": build_birnn,
    "LSTM":              build_lstm,
}

results = []

for feat_name, (X_tr, X_te) in feature_sets.items():
    for model_name, builder in model_builders.items():
        print(f"\n>>> {model_name}  +  {feat_name}")
        model = builder(X_tr.shape[1:])
        model.fit(X_tr, y_train,
                  validation_split=0.1,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  verbose=1)
        preds = (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='binary')
        results.append({
            "Feature":  feat_name,
            "Model":    model_name,
            "Accuracy": round(acc * 100, 2),
            "F1 Score": round(f1  * 100, 2),
        })
        print(f"  ✓  Acc: {acc:.4f}  |  F1: {f1:.4f}")

# ──────────────────────────────────────────
# 6. RESULTS TABLE
# ──────────────────────────────────────────
df_res = pd.DataFrame(results)

# Wide pivot (feature × model)
pivot = df_res.pivot_table(
    index="Feature", columns="Model",
    values=["Accuracy", "F1 Score"]
)
pivot.columns = [f"{col[1]} {col[0]}" for col in pivot.columns]
pivot = pivot.sort_values("LSTM Accuracy", ascending=False)

print("\n")
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║           IMDB SENTIMENT CLASSIFICATION — RESULTS TABLE             ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print(pivot.to_string())

# Also save
pivot.to_csv("imdb_results.csv")
print("\n✓ Saved to imdb_results.csv")