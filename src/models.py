import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        out = self.linear_layer(output[:, -1])
        return out
