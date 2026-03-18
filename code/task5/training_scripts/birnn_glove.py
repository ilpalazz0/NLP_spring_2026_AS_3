import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from data_cleaning import load_and_clean_data
from features.glove import load_glove_embeddings, get_glove_features, save_glove_embeddings
from models.BiRNN import BiRNNModel

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
GLOVE_PATH      = "./features/vectors_glove.txt"
SAMPLE_SIZE     = 10000
BATCH_SIZE      = 32
EPOCHS          = 100
HIDDEN_SIZE     = 64
INPUT_SIZE      = 300
LR              = 0.001
MAX_SEQ_LEN     = 100
MODEL_SAVE_PATH = "./saved_models/birnn_glove.pth"
EMBED_SAVE_PATH = "./saved_models/glove_embeddings.pkl"

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
print(f"\n[Embeddings] Loading GloVe from {GLOVE_PATH}...")
embeddings = load_glove_embeddings(GLOVE_PATH)
print(f"[Embeddings] Vocabulary size: {len(embeddings)}")

os.makedirs(os.path.dirname(EMBED_SAVE_PATH), exist_ok=True)
save_glove_embeddings(embeddings, EMBED_SAVE_PATH)
print(f"[Embeddings] Saved to {EMBED_SAVE_PATH}")

# ==========================
# Feature extraction
# ==========================
print("\n[Features] Converting text to GloVe sequences...")
X_train_vec, X_test_vec = get_glove_features(X_train, X_test, embeddings)

seq_lengths = [len(s) for s in X_train_vec]
print(f"[Features] Avg sequence length: {sum(seq_lengths)/len(seq_lengths):.0f} words")
print(f"[Features] Max sequence length: {max(seq_lengths)} words")
print(f"[Features] Min sequence length: {min(seq_lengths)} words")
print(f"[Features] Truncating to MAX_SEQ_LEN={MAX_SEQ_LEN}")

# ==========================
# Build tensors
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

print(f"[Tensors] X_train shape: {X_train_tensor.shape}")
print(f"[Tensors] X_test shape:  {X_test_tensor.shape}")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"[Tensors] Batches per epoch: {len(train_loader)}")

# ==========================
# Model
# ==========================
print(f"\n[Model] Initializing BiRNNModel(input_size={INPUT_SIZE}, hidden_size={HIDDEN_SIZE})")
model = BiRNNModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
model.to(device)
print(f"[Model] Model moved to: {next(model.parameters()).device}")
print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Sanity check
# ==========================
print("\n[Verify] Running sanity check on first batch...")
sample_X, sample_y = next(iter(train_loader))
sample_X = sample_X.to(device)
sample_y = sample_y.to(device)
print(f"[Verify] Batch X device: {sample_X.device}")
print(f"[Verify] Batch X shape:  {sample_X.shape}")
with torch.no_grad():
    sample_out = model(sample_X)
print(f"[Verify] Output shape:   {sample_out.shape}")
print("[Verify] Sanity check passed ✓")

# ==========================
# Training loop
# ==========================
print(f"\n[Training] Starting training for {EPOCHS} epochs...")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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