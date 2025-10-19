import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dense2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        score = self.dense2(torch.tanh(self.dense1(x)))
        weights = F.softmax(score, dim=1)
        context_vector = torch.sum(x * weights, dim=1)
        return context_vector

class AttentionLSTM(nn.Module):
    def __init__(self, num_features, lstm_units=150, dropout_rate=0.3):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(lstm_units * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        out = self.dropout(context_vector)
        out = self.fc(out)
        return out