import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from data_cleaning import load_and_clean_data
from models.RNN import RNNModel
from models.BiRNN import BiRNNModel
from models.LSTM import LSTMModel
from features.count_vectorizer import load_count_vectorizer
from features.tfidf import load_tfidf_vectorizer
from features.pmi import load_pmi_features, text_to_pmi_sequence
from features.word2vec import load_word2vec_embeddings_pkl, get_word2vec_features
from features.glove import load_glove_embeddings_pkl, get_glove_features

# ==========================
# Config
# ==========================
DATA_PATH        = "../../data/Sentiment/dataset.csv"
SAMPLE_SIZE      = 10000
SAVED_MODELS_DIR = "./saved_models/"
MAX_SEQ_LEN      = 100
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Device] Using: {device}")

EVAL_CONFIGS = [
    # (model_type, feature_type, hidden_size, num_layers, input_size)
    ("rnn",   "count",    32, 1, 1),
    ("rnn",   "tfidf",    32, 1, 1),
    ("rnn",   "pmi",      32, 1, 1),
    ("rnn",   "word2vec", 64, 1, 300),
    ("rnn",   "glove",    64, 1, 300),
    ("birnn", "count",    32, 1, 1),
    ("birnn", "tfidf",    32, 1, 1),
    ("birnn", "pmi",      32, 1, 1),
    ("birnn", "word2vec", 64, 1, 300),
    ("birnn", "glove",    64, 1, 300),
    ("lstm",  "count",    32, 1, 1),
    ("lstm",  "tfidf",    32, 1, 1),
    ("lstm",  "pmi",      32, 1, 1),
    ("lstm",  "word2vec", 64, 1, 300),
    ("lstm",  "glove",    64, 1, 300),
]

# ==========================
# Load test data
# ==========================
print("\n[Data] Loading test data...")
df = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
X_test = df["clean_review"][int(0.8 * len(df)):]
y_test = df["sentiment"][int(0.8 * len(df)):]
y_true = y_test.values
print(f"[Data] Test samples: {len(X_test)}")

print(f"First test review: {X_test.iloc[0][:50]}")
print(f"First test label:  {y_true[0]}")

# ==========================
# Load feature transformers
# ==========================
print("\n[Features] Loading all feature transformers...")

count_vectorizer = load_count_vectorizer(os.path.join(SAVED_MODELS_DIR, "count_vectorizer.pkl"))
tfidf_vectorizer = load_tfidf_vectorizer(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
pmi_scores       = load_pmi_features(os.path.join(SAVED_MODELS_DIR, "pmi_scores.pkl"))
w2v_embeddings   = load_word2vec_embeddings_pkl(os.path.join(SAVED_MODELS_DIR, "word2vec_embeddings.pkl"))
glove_embeddings = load_glove_embeddings_pkl(os.path.join(SAVED_MODELS_DIR, "glove_embeddings.pkl"))

print("[Features] Building test tensors...")

count_tensor = torch.tensor(
    count_vectorizer.transform(X_test).toarray(), dtype=torch.float32
).unsqueeze(2).to(device)

tfidf_tensor = torch.tensor(
    tfidf_vectorizer.transform(X_test).toarray(), dtype=torch.float32
).unsqueeze(2).to(device)

pmi_tensor = pad_sequence(
    [torch.tensor(text_to_pmi_sequence(t, pmi_scores), dtype=torch.float32) for t in X_test],
    batch_first=True
).unsqueeze(2).to(device)

_, w2v_test = get_word2vec_features(X_test, X_test, w2v_embeddings)
w2v_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in w2v_test],
    batch_first=True
).to(device)

_, glove_test = get_glove_features(X_test, X_test, glove_embeddings)
glove_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in glove_test],
    batch_first=True
).to(device)

feature_tensors = {
    "count":    count_tensor,
    "tfidf":    tfidf_tensor,
    "pmi":      pmi_tensor,
    "word2vec": w2v_tensor,
    "glove":    glove_tensor,
}

print("[Features] All test tensors ready ✓")

# ==========================
# Evaluate each model
# ==========================
print("\n[Evaluation] Running evaluation on all models...")
results = []

for model_type, feature_type, hidden_size, num_layers, input_size in EVAL_CONFIGS:
    model_file = os.path.join(SAVED_MODELS_DIR, f"{model_type}_{feature_type}.pth")

    if not os.path.exists(model_file):
        print(f"  [SKIP] {model_type}_{feature_type} — model file not found")
        continue

    if model_type == "rnn":
        model = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "birnn":
        model = BiRNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    elif model_type == "lstm":
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    X_tensor = feature_tensors[feature_type]

    all_preds = []
    EVAL_BATCH_SIZE = 64
    with torch.no_grad():
        for i in range(0, X_tensor.shape[0], EVAL_BATCH_SIZE):
            batch = X_tensor[i:i+EVAL_BATCH_SIZE].to(device)
            out = model(batch)
            all_preds.append((out >= 0.5).float().cpu())
    preds = torch.cat(all_preds).numpy()

    accuracy = accuracy_score(y_true, preds)
    f1       = f1_score(y_true, preds)

    results.append({
        "model":    model_type.upper(),
        "feature":  feature_type,
        "label":    f"{model_type.upper()}\n{feature_type}",
        "accuracy": accuracy,
        "f1":       f1,
    })

    print(f"  {model_type.upper():<6} + {feature_type:<8}  |  Accuracy: {accuracy:.4f}  |  F1: {f1:.4f}")

# ==========================
# Print summary table
# ==========================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<8} {'Feature':<10} {'Accuracy':>10} {'F1 Score':>10}")
print("-" * 42)
for r in results:
    print(f"{r['model']:<8} {r['feature']:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")

if results:
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest model: {best['model']} + {best['feature']}  →  Accuracy: {best['accuracy']:.4f}, F1: {best['f1']:.4f}")

# ==========================
# Plot results
# ==========================
if not results:
    print("\n[Plot] No results to plot.")
    sys.exit(0)

print("\n[Plot] Generating comparison chart...")

labels    = [r["label"]    for r in results]
accuracy  = [r["accuracy"] for r in results]
f1_scores = [r["f1"]       for r in results]
models    = [r["model"]    for r in results]

color_map  = {"RNN": "#4C9BE8", "BIRNN": "#E8834C", "LSTM": "#4CE87A"}
bar_colors = [color_map[m] for m in models]

x     = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(18, 7))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

bars1 = ax.bar(x - width/2, accuracy,  width, color=bar_colors, alpha=0.95)
bars2 = ax.bar(x + width/2, f1_scores, width, color=bar_colors, alpha=0.50)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=7.5, color="white", fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=7.5, color="#cccccc")

ax.axhline(y=0.5, color="#ff4444", linestyle="--", linewidth=1.2, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, color="white")
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Score", color="white", fontsize=11)
ax.set_title("Model Performance Comparison — Accuracy & F1 Score",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.yaxis.set_tick_params(labelcolor="white")
ax.grid(axis="y", color="#333333", linewidth=0.7, alpha=0.6)

legend_patches = [
    mpatches.Patch(color=color_map["RNN"],   label="RNN"),
    mpatches.Patch(color=color_map["BIRNN"], label="BiRNN"),
    mpatches.Patch(color=color_map["LSTM"],  label="LSTM"),
    mpatches.Patch(color="gray", alpha=0.95, label="Accuracy (solid)"),
    mpatches.Patch(color="gray", alpha=0.45, label="F1 Score (faded)"),
    mpatches.Patch(color="#ff4444",          label="Random chance (0.5)"),
]
ax.legend(handles=legend_patches, facecolor="#1a1a2e", edgecolor="#333333",
          labelcolor="white", fontsize=9, loc="upper left")

plt.tight_layout()
output_path = "./results_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[Plot] Chart saved to {output_path}")
print("\nDone.")