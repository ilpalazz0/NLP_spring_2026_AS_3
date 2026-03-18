from collections import Counter

class SimpleTokenizer:
    def __init__(self, max_words=10000):
        self.max_words = max_words
        self.word_index = {}
        self.index_word = {}

    def fit(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(text.split())

        most_common = counter.most_common(self.max_words - 2)

        # Reserve 0 for padding, 1 for OOV
        self.word_index = {"<PAD>": 0, "<OOV>": 1}

        for i, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = i

        self.index_word = {i: w for w, i in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []

        for text in texts:
            seq = [
                self.word_index.get(word, 1)  # 1 = OOV
                for word in text.split()
            ]
            sequences.append(seq)

        return sequences


def pad_sequences(sequences, max_len=200):
    padded = []

    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        padded.append(seq)

    return padded


def get_tokenized_data(X_train, X_test, max_words=10000, max_len=200):
    tokenizer = SimpleTokenizer(max_words=max_words)
    tokenizer.fit(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, max_len)
    X_test_pad = pad_sequences(X_test_seq, max_len)

    return X_train_pad, X_test_pad, tokenizer