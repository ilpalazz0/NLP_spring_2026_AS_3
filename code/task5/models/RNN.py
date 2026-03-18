import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.3 if num_layers > 1 else 0
    )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        out = self.fc(out)
        return self.sigmoid(out)