import torch
import torch.nn as nn


class BiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout_rate=0.4):
        super(BiRNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        # hidden_size * 2 because we concatenate forward + backward last states
        self.fc      = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out:    (batch, seq_len, hidden_size * 2)
        # hidden: (num_layers * 2, batch, hidden_size)
        _, hidden = self.rnn(x)

        # For a single-layer BiRNN:
        #   hidden[0] = last forward  state  (batch, hidden_size)
        #   hidden[1] = last backward state  (batch, hidden_size)
        fwd = hidden[0]
        bwd = hidden[1]

        out = torch.cat([fwd, bwd], dim=1)  # (batch, hidden_size * 2)
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)