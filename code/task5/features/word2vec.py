import numpy as np
import pickle


EMBEDDING_DIM = 300


def load_word2vec_embeddings(path):
    """
    Load Word2Vec embeddings from a .txt file.
    Format: first line is "vocab_size dim" header,
            then each line is "word v1 v2 ... v300"
    Returns a dict: {word: np.array of shape (300,)}
    """
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split()

        # If first line is "vocab_size dim" header, skip it
        # Otherwise it's a word vector — parse it
        if len(first_line) == 2 and first_line[1].isdigit():
            pass  # header consumed, move on
        else:
            word = first_line[0]
            vector = np.array(first_line[1:], dtype=np.float32)
            if len(vector) == EMBEDDING_DIM:
                embeddings[word] = vector

        for line in f:
            parts = line.strip().split()
            if len(parts) != EMBEDDING_DIM + 1:
                continue
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector

    print(f"Loaded {len(embeddings)} Word2Vec vectors from {path}")
    return embeddings


def text_to_word2vec_sequence(text, embeddings):
    """
    Convert a cleaned text string into an array of 300-dim vectors,
    one per word. Unknown words are replaced with a zero vector.
    Returns np.array of shape (num_words, 300).
    """
    words = text.split()
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vectors = [embeddings.get(word, zero) for word in words]
    if len(vectors) == 0:
        vectors.append(zero)
    return np.array(vectors, dtype=np.float32)


def get_word2vec_features(X_train, X_test, embeddings):
    """
    Convert train and test text series into lists of np.arrays.
    Each element is shape (num_words, 300).
    """
    X_train_vec = [text_to_word2vec_sequence(t, embeddings) for t in X_train]
    X_test_vec  = [text_to_word2vec_sequence(t, embeddings) for t in X_test]
    return X_train_vec, X_test_vec


def save_word2vec_embeddings(embeddings, path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_word2vec_embeddings_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)