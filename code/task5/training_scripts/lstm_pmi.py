import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_cleaning import load_and_clean_data
from features.pmi import get_pmi_features
from models.LSTM import LSTMModel
import pickle
import os
from torch.nn.utils.rnn import pad_sequence

# Config
DATA_PATH = "../../data/Sentiment/dataset.csv"
SAMPLE_SIZE = 10000
BATCH_SIZE = 32
EPOCHS = 5
HIDDEN_SIZE = 32
LR = 0.001
MODEL_SAVE_PATH = "./saved_models/lstm_pmi.pth"
PMI_SAVE_PATH = "./saved_models/pmi_scores.pkl"

# Load data
df = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
X_train = df["clean_review"][:int(0.8*len(df))]
X_test  = df["clean_review"][int(0.8*len(df)):]
y_train = df["sentiment"][:int(0.8*len(df))]
y_test  = df["sentiment"][int(0.8*len(df)):]

# PMI features
X_train_vec, X_test_vec, pmi_scores = get_pmi_features(X_train, X_test, vocab_size=5000)

# Save PMI scores
os.makedirs(os.path.dirname(PMI_SAVE_PATH), exist_ok=True)
with open(PMI_SAVE_PATH, "wb") as f:
    pickle.dump(pmi_scores, f)

# Pad sequences and convert to tensors
X_train_tensor = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in X_train_vec], batch_first=True).unsqueeze(2)
X_test_tensor  = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in X_test_vec], batch_first=True).unsqueeze(2)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
input_size = 1
model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_SIZE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved at", MODEL_SAVE_PATH)