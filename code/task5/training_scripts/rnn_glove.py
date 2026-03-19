import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from data_cleaning import load_and_clean_data
from features.glove import load_glove_embeddings, save_glove_embeddings
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
GLOVE_PATH      = "./features/vectors_glove.txt"
SAMPLE_SIZE     = 100000
BATCH_SIZE      = 64
EPOCHS          = 50
HIDDEN_SIZE     = 128
INPUT_SIZE      = 300
LR              = 0.001
MAX_NORM        = 1.0
MAX_SEQ_LEN     = 100
PATIENCE        = 7
MODEL_SAVE_PATH = "./saved_models/rnn_glove.pth"
EMBED_SAVE_PATH = "./saved_models/glove_embeddings.pkl"
LOG_SAVE_PATH   = "./code/task5/loss_points/rnn_glove.txt"

# ==========================
# Load & clean data
# ==========================
print("\n[Data] Loading and cleaning data...")
df = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
print(f"[Data] Total samples: {len(df)}")
print(f"[Data] Label distribution:\n{df['sentiment'].value_counts().to_string()}")

n       = len(df)
n_train = int(0.72 * n)
n_val   = int(0.08 * n)

X_train = df["clean_review"][:n_train].reset_index(drop=True)
X_val   = df["clean_review"][n_train:n_train + n_val].reset_index(drop=True)
X_test  = df["clean_review"][n_train + n_val:].reset_index(drop=True)
y_train = df["sentiment"][:n_train].reset_index(drop=True)
y_val   = df["sentiment"][n_train:n_train + n_val].reset_index(drop=True)
y_test  = df["sentiment"][n_train + n_val:].reset_index(drop=True)
print(f"[Data] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ==========================
# Load embeddings
# ==========================
print(f"\n[Embeddings] Loading GloVe from {GLOVE_PATH}...")
embeddings = load_glove_embeddings(GLOVE_PATH)
print(f"[Embeddings] Vocabulary size: {len(embeddings)}")

os.makedirs(os.path.dirname(EMBED_SAVE_PATH), exist_ok=True)
save_glove_embeddings(embeddings, EMBED_SAVE_PATH)
print(f"[Embeddings] Saved to {EMBED_SAVE_PATH}")

# Pre-compute OOV vector once (mean of all vectors)
oov_vector = np.mean(list(embeddings.values()), axis=0).astype(np.float32)
print(f"[Embeddings] OOV fallback: mean vector computed")

# ==========================
# Dataset — vectors looked up on the fly, no pre-allocation
# ==========================
class SequenceDataset(Dataset):
    def __init__(self, texts, labels, embeddings, oov_vector, max_len):
        self.texts      = texts
        self.labels     = labels
        self.embeddings = embeddings
        self.oov        = oov_vector
        self.max_len    = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()[:self.max_len]
        vecs  = [self.embeddings.get(w, self.oov) for w in words]
        if len(vecs) == 0:
            vecs = [self.oov]
        x = torch.tensor(np.array(vecs, dtype=np.float32))
        y = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded  = pad_sequence(xs, batch_first=True)
    ys_stacked = torch.stack(ys).unsqueeze(1)
    return xs_padded, ys_stacked


# ==========================
# DataLoaders
# ==========================
print("\n[DataLoader] Building datasets...")
train_dataset = SequenceDataset(X_train, y_train, embeddings, oov_vector, MAX_SEQ_LEN)
val_dataset   = SequenceDataset(X_val,   y_val,   embeddings, oov_vector, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)
print(f"[DataLoader] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ==========================
# Model
# ==========================
print(f"\n[Model] Initializing RNNModel(input_size={INPUT_SIZE}, hidden_size={HIDDEN_SIZE})")
model = RNNModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
model.to(device)
print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# ==========================
# Sanity check
# ==========================
print("\n[Verify] Running sanity check on first batch...")
sample_X, sample_y = next(iter(train_loader))
sample_X = sample_X.to(device)
print(f"[Verify] Batch X shape: {sample_X.shape}")
with torch.no_grad():
    sample_out = model(sample_X)
print(f"[Verify] Output shape:  {sample_out.shape}")
print("[Verify] Sanity check passed ✓")

# ==========================
# Logging setup
# ==========================
os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)
log_file = open(LOG_SAVE_PATH, "w")
log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

# ==========================
# Early stopping state
# ==========================
best_val_loss    = float('inf')
patience_counter = 0
best_epoch       = 0

# ==========================
# Training loop
# ==========================
print(f"\n[Training] Starting training for up to {EPOCHS} epochs (patience={PATIENCE})...")

for epoch in range(EPOCHS):
    # ---- train ----
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)

        if not torch.isfinite(loss):
            print(f"  [WARNING] Non-finite loss at epoch {epoch+1}, skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
        optimizer.step()
        total_loss += loss.item()
        preds       = (outputs >= 0.5).float()
        correct    += (preds == y_batch).sum().item()
        total      += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total if total > 0 else 0.0

    # ---- validation ----
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch      = X_batch.to(device)
            y_batch      = y_batch.to(device)
            outputs      = model(X_batch)
            val_loss_sum += criterion(outputs, y_batch).item()
            preds        = (outputs >= 0.5).float()
            val_correct  += (preds == y_batch).sum().item()
            val_total    += y_batch.size(0)

    val_loss = val_loss_sum / len(val_loader)
    val_acc  = val_correct / val_total if val_total > 0 else 0.0

    scheduler.step(val_loss)

    vram_info = f"  |  VRAM: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB" if device.type == "cuda" else ""
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"| Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}"
          f"  | Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
          f"{vram_info}")

    log_file.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")
    log_file.flush()

    # ---- early stopping ----
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_epoch       = epoch + 1
        patience_counter = 0
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  [Checkpoint] Best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  [EarlyStopping] No improvement for {patience_counter}/{PATIENCE} epochs.")
        if patience_counter >= PATIENCE:
            print(f"\n[Early Stop] Triggered at epoch {epoch+1}. Best epoch was {best_epoch}.")
            break

log_file.close()
print(f"\n[Log] Metrics saved to {LOG_SAVE_PATH}")
print(f"[Best] val_loss={best_val_loss:.4f} at epoch {best_epoch}")
print(f"[Saved] Model at {MODEL_SAVE_PATH}")