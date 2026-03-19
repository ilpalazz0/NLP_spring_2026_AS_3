import numpy as np
import pickle


EMBEDDING_DIM = 300


def _is_header(parts):
    """
    Return True if the line looks like a Word2Vec header
    e.g. "88091 300" — two tokens, both purely numeric integers.
    This is more robust than just checking parts[1].isdigit().
    """
    return (
        len(parts) == 2
        and parts[0].isdigit()
        and parts[1].isdigit()
    )


def load_word2vec_embeddings(path, normalize=True):
    """
    Load Word2Vec embeddings from a .txt file.
    Format: optional first line "vocab_size dim" header,
            then each line is "word v1 v2 ... v300".
    Returns a dict: {word: np.array of shape (300,)}

    normalize=True rescales every vector to unit length, which
    stabilises training when raw magnitudes are very small.
    """
    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split()

        # Only skip the line if it is genuinely a numeric header
        if not _is_header(first_line):
            # Not a header — parse it as a word vector
            if len(first_line) == EMBEDDING_DIM + 1:
                word   = first_line[0]
                vector = np.array(first_line[1:], dtype=np.float32)
                if normalize:
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                embeddings[word] = vector

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

    print(f"[Word2Vec] Loaded {len(embeddings)} vectors from {path}"
          f"  (normalize={normalize})")
    return embeddings


def _mean_vector(embeddings):
    """Return the mean of all embedding vectors (used as OOV fallback)."""
    if not embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.mean(list(embeddings.values()), axis=0).astype(np.float32)


def text_to_word2vec_sequence(text, embeddings, oov_vector):
    """
    Convert a cleaned text string into an array of 300-dim vectors,
    one per word.  Unknown words use oov_vector instead of zeros.
    Returns np.array of shape (num_words, 300).
    """
    words   = text.split()
    vectors = [embeddings.get(word, oov_vector) for word in words]
    if len(vectors) == 0:
        vectors.append(oov_vector)
    return np.array(vectors, dtype=np.float32)


def get_word2vec_features(X_train, X_other, embeddings):
    """
    Convert train and one other split (val or test) into lists of
    np.arrays.  OOV words are replaced with the mean training vector
    so the fallback carries meaningful signal rather than dead zeros.

    Each element is shape (num_words, 300).
    """
    oov = _mean_vector(embeddings)
    X_train_vec = [text_to_word2vec_sequence(t, embeddings, oov) for t in X_train]
    X_other_vec = [text_to_word2vec_sequence(t, embeddings, oov) for t in X_other]
    return X_train_vec, X_other_vec


def save_word2vec_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_word2vec_embeddings_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)