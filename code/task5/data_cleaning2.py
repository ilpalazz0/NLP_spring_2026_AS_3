import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ll", "'ll").replace(" ve", "'ve")
    return text


def load_and_clean_data(path, sample_size=None):
    df = pd.read_csv(path)

    # Drop missing
    df = df.dropna(subset=["review", "sentiment"])

    # Normalize labels
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    # Clean text
    df["clean_review"] = df["review"].apply(clean_text)

    # Optional sampling
    if sample_size:
        df = df.sample(sample_size, random_state=42)

    return df