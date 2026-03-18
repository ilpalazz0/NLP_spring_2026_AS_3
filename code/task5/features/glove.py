import numpy as np
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch


EMBEDDING_DIM = 300


def load_glove_embeddings(txt_path):
    """
    Load GloVe embeddings from a .txt file.
    Format: each line is "word v1 v2 ... v300" (no header line).
    Returns a dict: {word: np.array of shape (300,)}
    """
    embeddings = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != EMBEDDING_DIM + 1:
                continue
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} GloVe vectors from {txt_path}")
    return embeddings


def text_to_glove_sequence(text, embeddings):
    """
    Convert a cleaned text string into a list of 300-dim vectors,
    one per word. Unknown words are replaced with a zero vector.
    Returns np.array of shape (num_words, 300).
    """
    words = text.split()
    vectors = []
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for word in words:
        vectors.append(embeddings.get(word, zero))
    if len(vectors) == 0:
        vectors.append(zero)  # avoid empty sequence
    return np.array(vectors, dtype=np.float32)


def get_glove_features(X_train, X_test, embeddings):
    """
    Convert train and test text series into lists of np.arrays.
    Each element is shape (num_words, 300).
    """
    X_train_vec = [text_to_glove_sequence(t, embeddings) for t in X_train]
    X_test_vec  = [text_to_glove_sequence(t, embeddings) for t in X_test]
    return X_train_vec, X_test_vec


def save_glove_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_glove_embeddings_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)