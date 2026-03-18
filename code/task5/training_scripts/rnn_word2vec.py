import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from data_cleaning import load_and_clean_data
from features.word2vec import load_word2vec_embeddings, get_word2vec_features, save_word2vec_embeddings
from models.RNN import RNNModel

# ==========================
# Device check
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")
if device.type == "cuda":
    print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Device] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[Device] WARNING: GPU not found, running on CPU")

# ==========================
# Config
# ==========================
DATA_PATH       = "../../data/Sentiment/dataset.csv"
W2V_PATH        = "./features/vectors_w2vec.txt"
SAMPLE_SIZE     = 10000
BATCH_SIZE      = 32
EPOCHS          = 100
HIDDEN_SIZE     = 64
INPUT_SIZE      = 300
LR              = 0.001
MODEL_SAVE_PATH = "./saved_models/rnn_word2vec.pth"
EMBED_SAVE_PATH = "./saved_models/word2vec_embeddings.pkl"
MAX_SEQ_LEN = 100

# ==========================
# Load & clean data
# ==========================
print("\n[Data] Loading and cleaning data...")
df = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
print(f"[Data] Total samples: {len(df)}")
print(f"[Data] Label distribution:\n{df['sentiment'].value_counts().to_string()}")

X_train = df["clean_review"][:int(0.8 * len(df))]
X_test  = df["clean_review"][int(0.8 * len(df)):]
y_train = df["sentiment"][:int(0.8 * len(df))]
y_test  = df["sentiment"][int(0.8 * len(df)):]
print(f"[Data] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ==========================
# Load embeddings
# ==========================
print(f"\n[Embeddings] Loading Word2Vec from {W2V_PATH}...")
embeddings = load_word2vec_embeddings(W2V_PATH)
print(f"[Embeddings] Vocabulary size: {len(embeddings)}")

os.makedirs(os.path.dirname(EMBED_SAVE_PATH), exist_ok=True)
save_word2vec_embeddings(embeddings, EMBED_SAVE_PATH)
print(f"[Embeddings] Saved to {EMBED_SAVE_PATH}")

# ==========================
# Feature extraction
# ==========================
print("\n[Features] Converting text to Word2Vec sequences...")
X_train_vec, X_test_vec = get_word2vec_features(X_train, X_test, embeddings)

seq_lengths = [len(s) for s in X_train_vec]
print(f"[Features] Avg sequence length: {sum(seq_lengths)/len(seq_lengths):.0f} words")
print(f"[Features] Max sequence length: {max(seq_lengths)} words")
print(f"[Features] Min sequence length: {min(seq_lengths)} words")

# ==========================
# Build tensors — kept on CPU, moved to GPU per batch
# ==========================
print("\n[Tensors] Building padded tensors...")
X_train_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in X_train_vec],
    batch_first=True)
X_test_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in X_test_vec],
    batch_first=True)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

print(f"[Tensors] X_train shape: {X_train_tensor.shape}")  # (8000, max_seq_len, 300)
print(f"[Tensors] X_test shape:  {X_test_tensor.shape}")
print(f"[Tensors] y_train shape: {y_train_tensor.shape}")
print(f"[Tensors] Tensors on: CPU (will move to {device} per batch)")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"[Tensors] Batches per epoch: {len(train_loader)}")

# ==========================
# Model
# ==========================
print(f"\n[Model] Initializing RNNModel(input_size={INPUT_SIZE}, hidden_size={HIDDEN_SIZE})")
model = RNNModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
model.to(device)
print(f"[Model] Model moved to: {next(model.parameters()).device}")
print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Sanity check — verify GPU is actually used
# ==========================
print("\n[Verify] Running sanity check on first batch...")
sample_X, sample_y = next(iter(train_loader))
sample_X = sample_X.to(device)
sample_y = sample_y.to(device)
print(f"[Verify] Batch X device: {sample_X.device}")
print(f"[Verify] Batch y device: {sample_y.device}")
print(f"[Verify] Batch X shape:  {sample_X.shape}")  # (32, seq_len, 300)
with torch.no_grad():
    sample_out = model(sample_X)
print(f"[Verify] Output shape:   {sample_out.shape}")  # (32, 1)
print(f"[Verify] Output device:  {sample_out.device}")
print("[Verify] Sanity check passed ✓")

# ==========================
# Training loop
# ==========================
print(f"\n[Training] Starting training for {EPOCHS} epochs...")
if device.type == "cuda":
    print(f"[Training] VRAM used before training: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    vram_info = f"  |  VRAM: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB" if device.type == "cuda" else ""
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}{vram_info}")

# ==========================
# Save model
# ==========================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n[Saved] Model saved at {MODEL_SAVE_PATH}")