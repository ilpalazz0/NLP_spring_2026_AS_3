import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from data_cleaning import clean_text
from models.RNN import RNNModel
from models.BiRNN import BiRNNModel
from models.LSTM import LSTMModel
from features.count_vectorizer import load_count_vectorizer
from features.tfidf import load_tfidf_vectorizer
from features.pmi import load_pmi_features, text_to_pmi_sequence
from features.word2vec import load_word2vec_embeddings_pkl, text_to_word2vec_sequence
from features.glove import load_glove_embeddings_pkl, text_to_glove_sequence
from torch.nn.utils.rnn import pad_sequence

# ==========================
# Config: choose your model
# ==========================
MODEL_TYPE   = "lstm"      # options: "rnn", "birnn", "lstm"
FEATURE_TYPE = "glove"     # options: "count", "tfidf", "pmi", "word2vec", "glove"

SAVED_MODELS_DIR = "./saved_models/"
MAX_SEQ_LEN      = 100     # must match training

# ==========================
# Architecture config
# must match what was used during training
# ==========================
if FEATURE_TYPE in ["word2vec", "glove"]:
    HIDDEN_SIZE = 64
    INPUT_SIZE  = 300
else:
    HIDDEN_SIZE = 32
    INPUT_SIZE  = 1

# ==========================
# PMI/vectorizer file path
# ==========================
if FEATURE_TYPE == "pmi":
    VECTOR_FILE = os.path.join(SAVED_MODELS_DIR, "pmi_scores.pkl")
elif FEATURE_TYPE == "word2vec":
    VECTOR_FILE = os.path.join(SAVED_MODELS_DIR, "word2vec_embeddings.pkl")
elif FEATURE_TYPE == "glove":
    VECTOR_FILE = os.path.join(SAVED_MODELS_DIR, "glove_embeddings.pkl")
else:
    VECTOR_FILE = os.path.join(SAVED_MODELS_DIR, f"{FEATURE_TYPE}_vectorizer.pkl")

MODEL_FILE = os.path.join(SAVED_MODELS_DIR, f"{MODEL_TYPE}_{FEATURE_TYPE}.pth")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")

# ==========================
# Load model
# ==========================
if MODEL_TYPE == "rnn":
    model = RNNModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
elif MODEL_TYPE == "birnn":
    model = BiRNNModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
elif MODEL_TYPE == "lstm":
    model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
else:
    raise ValueError("MODEL_TYPE must be 'rnn', 'birnn', or 'lstm'")

model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.to(device)
model.eval()
print(f"[Model] Loaded {MODEL_TYPE}_{FEATURE_TYPE}")

# ==========================
# Load feature transformer
# ==========================
if FEATURE_TYPE == "count":
    vectorizer = load_count_vectorizer(VECTOR_FILE)
elif FEATURE_TYPE == "tfidf":
    vectorizer = load_tfidf_vectorizer(VECTOR_FILE)
elif FEATURE_TYPE == "pmi":
    pmi_scores = load_pmi_features(VECTOR_FILE)
elif FEATURE_TYPE == "word2vec":
    embeddings = load_word2vec_embeddings_pkl(VECTOR_FILE)
elif FEATURE_TYPE == "glove":
    embeddings = load_glove_embeddings_pkl(VECTOR_FILE)
else:
    raise ValueError("Unknown FEATURE_TYPE")

print(f"[Features] Loaded {FEATURE_TYPE} transformer")

# ==========================
# Prediction function
# ==========================
def predict_sentiment(text: str):
    cleaned = clean_text(text)

    if FEATURE_TYPE in ["count", "tfidf"]:
        X_vec    = vectorizer.transform([cleaned])
        X_tensor = torch.tensor(X_vec.toarray(), dtype=torch.float32).unsqueeze(2).to(device)

    elif FEATURE_TYPE == "pmi":
        seq      = text_to_pmi_sequence(cleaned, pmi_scores)
        X_tensor = pad_sequence(
            [torch.tensor(seq, dtype=torch.float32)],
            batch_first=True
        ).unsqueeze(2).to(device)

    elif FEATURE_TYPE == "word2vec":
        seq      = text_to_word2vec_sequence(cleaned, embeddings)
        X_tensor = pad_sequence(
            [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)],
            batch_first=True
        ).to(device)

    elif FEATURE_TYPE == "glove":
        seq      = text_to_glove_sequence(cleaned, embeddings)
        X_tensor = pad_sequence(
            [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32)],
            batch_first=True
        ).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        prob   = output.item()
        label  = "positive" if prob >= 0.5 else "negative"

    return label, prob


# ==========================
# Interactive loop
# ==========================
if __name__ == "__main__":
    print(f"\nReady — using {MODEL_TYPE.upper()} + {FEATURE_TYPE}")
    print("Enter an Azerbaijani review to classify.\n")
    while True:
        text = input("Review (or 'exit' to quit):\n> ")
        if text.lower() == "exit":
            break
        label, prob = predict_sentiment(text)
        print(f"Sentiment: {label}  (confidence: {prob:.4f})\n")