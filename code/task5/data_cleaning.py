import pandas as pd
import re

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep Azerbaijani letters (Latin script: a-z + ə, ı, ö, ü, ğ, ş, ç)
    # Also keep Cyrillic in case your dataset uses old Azerbaijani Cyrillic script
    text = re.sub(r"[^a-zəıöüğşç\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_clean_data(path, sample_size=None, text_col="review", label_col="sentiment"):
    df = pd.read_csv(path)

    df = df.dropna(subset=[text_col, label_col])

    # Normalize labels
    label_col_raw = df[label_col].astype(str).str.lower().str.strip()

    # Handle text labels (English)
    text_map = {"positive": 1, "negative": 0}

    def parse_label(val):
        if val in text_map:
            return text_map[val]
        try:
            return int(float(val))  # handles "1.0", "0.0", "1", "0"
        except ValueError:
            return None

    df[label_col] = label_col_raw.apply(parse_label)

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    df["clean_review"] = df[text_col].apply(clean_text)
    df = df[df["clean_review"].str.strip() != ""]

    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)

    return df