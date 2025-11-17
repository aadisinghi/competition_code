import torch 
import torch.nn as nn 
import torch.nn.functional as F
import pandas as pd 
import numpy as np
# -----------------------------
# 1) Simple LSTM Model
# -----------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=10, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, seq_len, hidden)
        out = self.fc(out[:, -1, :])   # use last hidden state
        return out  
    