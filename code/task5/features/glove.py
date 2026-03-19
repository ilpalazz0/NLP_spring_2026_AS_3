import numpy as np
import pickle


EMBEDDING_DIM = 300


def load_glove_embeddings(txt_path, normalize=True):
    """
    Load GloVe embeddings from a .txt file.
    Format: each line is "word v1 v2 ... v300" (no header line).
    Returns a dict: {word: np.array of shape (300,)}

    normalize=True rescales every vector to unit length, which
    stabilises training when raw magnitudes are very small.
    """
    embeddings = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != EMBEDDING_DIM + 1:
                continue
            word   = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            embeddings[word] = vector

    print(f"[GloVe] Loaded {len(embeddings)} vectors from {txt_path}"
          f"  (normalize={normalize})")
    return embeddings


def _mean_vector(embeddings):
    """Return the mean of all embedding vectors (used as OOV fallback)."""
    if not embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.mean(list(embeddings.values()), axis=0).astype(np.float32)


def text_to_glove_sequence(text, embeddings, oov_vector):
    """
    Convert a cleaned text string into a list of 300-dim vectors,
    one per word.  Unknown words use oov_vector instead of zeros.
    Returns np.array of shape (num_words, 300).
    """
    words   = text.split()
    vectors = [embeddings.get(word, oov_vector) for word in words]
    if len(vectors) == 0:
        vectors.append(oov_vector)
    return np.array(vectors, dtype=np.float32)


def get_glove_features(X_train, X_other, embeddings):
    """
    Convert train and one other split (val or test) into lists of
    np.arrays.  OOV words are replaced with the mean training vector
    so the fallback carries meaningful signal rather than dead zeros.

    Each element is shape (num_words, 300).
    """
    oov = _mean_vector(embeddings)
    X_train_vec = [text_to_glove_sequence(t, embeddings, oov) for t in X_train]
    X_other_vec = [text_to_glove_sequence(t, embeddings, oov) for t in X_other]
    return X_train_vec, X_other_vec


def save_glove_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_glove_embeddings_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)