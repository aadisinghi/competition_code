# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """
    Input:  x of shape (B, 60, 2)  -> 60 time steps, 2 features (close, volume)
    Output: next-10 values (B, 10)
    Design: Conv1d over time (extract local temporal patterns) -> LSTM (sequence modeling)
    """
    def __init__(self):
        super().__init__()
        # --- CNN over time ----------------------------------------------------
        # in_channels = #features
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)  # 60 -> 30 time steps

        # --- LSTM over the reduced sequence ----------------------------------
        # input_size to LSTM == conv feature channels (64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.2, bidirectional=False)

        # --- Regression head --------------------------------------------------
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        # x: (B, T, C) -> transpose for Conv1d which expects (B, C, T)
        x = x.transpose(1, 2)                  # (B, 2, 60)
        x = F.relu(self.conv1(x))              # (B, 32, 60)
        x = F.relu(self.conv2(x))              # (B, 64, 60)
        x = self.pool(x)                       # (B, 64, 30)

        # back to (B, T, C) for LSTM
        x = x.transpose(1, 2)                  # (B, 30, 64)
        out, _ = self.lstm(x)                  # (B, 30, 64)
        last = out[:, -1, :]                   # (B, 64)
        return self.head(last)                 # (B, 10)


def init_model(weights_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM().to(device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model
