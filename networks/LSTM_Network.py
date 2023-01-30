import torch
import torch.nn as nn


class LSTM_Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50000, 300)
        self.lstm = nn.LSTM(300, 2000, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return x



