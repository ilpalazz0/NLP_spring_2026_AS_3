import numpy as np
from collections import Counter
import math
import pickle

def load_pmi_features(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def text_to_pmi_sequence(text, pmi_scores):
    words = text.split()
    vec = []
    for i in range(len(words) - 1):
        pair = (words[i], words[i+1])
        vec.append(pmi_scores.get(pair, 0.0))
    return vec

def compute_pmi(corpus, vocab_size=5000):
    word_counts = Counter()
    pair_counts = Counter()
    total_words = 0

    for text in corpus:
        words = text.split()
        total_words += len(words)
        word_counts.update(words)

        for i in range(len(words) - 1):
            pair = (words[i], words[i+1])
            pair_counts[pair] += 1

    # Select top vocab
    vocab = dict(word_counts.most_common(vocab_size))

    pmi_scores = {}

    for (w1, w2), pair_count in pair_counts.items():
        if w1 in vocab and w2 in vocab:
            p_w1 = word_counts[w1] / total_words
            p_w2 = word_counts[w2] / total_words
            p_w1_w2 = pair_count / total_words

            pmi = math.log((p_w1_w2 / (p_w1 * p_w2)) + 1e-9)
            pmi_scores[(w1, w2)] = pmi

    return pmi_scores


def get_pmi_features(X_train, X_test, vocab_size=5000):
    pmi_scores = compute_pmi(X_train, vocab_size)

    def vectorize(text):
        words = text.split()
        vec = []

        for i in range(len(words) - 1):
            pair = (words[i], words[i+1])
            vec.append(pmi_scores.get(pair, 0))

        return np.array(vec)

    X_train_vec = [vectorize(t) for t in X_train]
    X_test_vec = [vectorize(t) for t in X_test]

    return X_train_vec, X_test_vec, pmi_scores