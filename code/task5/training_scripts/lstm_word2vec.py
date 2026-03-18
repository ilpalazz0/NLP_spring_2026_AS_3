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
from models.LSTM import LSTMModel
import os

DATA_PATH       = "../../data/Sentiment/dataset.csv"
W2V_PATH        = "./features/vectors_w2vec.txt"
SAMPLE_SIZE     = 10000
BATCH_SIZE      = 32
EPOCHS          = 100
HIDDEN_SIZE     = 64
INPUT_SIZE      = 300
LR              = 0.001
MODEL_SAVE_PATH = "./saved_models/lstm_word2vec.pth"
EMBED_SAVE_PATH = "./saved_models/word2vec_embeddings.pkl"
MAX_SEQ_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_and_clean_data(DATA_PATH, sample_size=SAMPLE_SIZE)
X_train = df["clean_review"][:int(0.8 * len(df))]
X_test  = df["clean_review"][int(0.8 * len(df)):]
y_train = df["sentiment"][:int(0.8 * len(df))]
y_test  = df["sentiment"][int(0.8 * len(df)):]

embeddings = load_word2vec_embeddings(W2V_PATH)

os.makedirs(os.path.dirname(EMBED_SAVE_PATH), exist_ok=True)
save_word2vec_embeddings(embeddings, EMBED_SAVE_PATH)

X_train_vec, X_test_vec = get_word2vec_features(X_train, X_test, embeddings)

X_train_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in X_train_vec],
    batch_first=True)
X_test_tensor = pad_sequence(
    [torch.tensor(seq[:MAX_SEQ_LEN], dtype=torch.float32) for seq in X_test_vec],
    batch_first=True)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved at", MODEL_SAVE_PATH)