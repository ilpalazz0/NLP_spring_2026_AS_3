from sklearn.feature_extraction.text import CountVectorizer
import pickle

def load_count_vectorizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_count_features(X_train, X_test, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer

