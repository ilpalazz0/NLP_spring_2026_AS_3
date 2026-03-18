import torch
import sys
sys.path.insert(0, '.')
from models.LSTM import LSTMModel
from features.word2vec import load_word2vec_embeddings_pkl, get_word2vec_features
from data_cleaning import load_and_clean_data
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_and_clean_data('../../data/Sentiment/dataset.csv', sample_size=100)
X = df['clean_review']

embeddings = load_word2vec_embeddings_pkl('./saved_models/word2vec_embeddings.pkl')
_, X_vec = get_word2vec_features(X, X, embeddings)
X_tensor = pad_sequence(
    [torch.tensor(seq[:100], dtype=torch.float32) for seq in X_vec],
    batch_first=True
).to(device)

model = LSTMModel(input_size=300, hidden_size=64)
model.load_state_dict(torch.load('./saved_models/lstm_word2vec.pth', map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    outputs = model(X_tensor)

print(f"Min output:  {outputs.min().item():.4f}")
print(f"Max output:  {outputs.max().item():.4f}")
print(f"Mean output: {outputs.mean().item():.4f}")
print(f"Sample outputs: {outputs[:10].squeeze().tolist()}")