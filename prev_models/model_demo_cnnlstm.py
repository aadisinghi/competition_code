# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid forecaster.
    Input:  (B, 60, 2) -> 60 time steps, 2 features (close, volume)
    Output: (B, 10)    -> next-10 predicted log prices
    """
    def __init__(self):
        super().__init__()
        # --- CNN over time ---
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)  # 60 -> 30

        # --- LSTM sequence model ---
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

        # --- Regression head ---
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        # x: (B, 60, 2)
        x = x.transpose(1, 2)        # -> (B, 2, 60)
        x = F.relu(self.conv1(x))    # -> (B, 32, 60)
        x = F.relu(self.conv2(x))    # -> (B, 64, 60)
        x = self.pool(x)             # -> (B, 64, 30)
        x = x.transpose(1, 2)        # -> (B, 30, 64)
        out, _ = self.lstm(x)        # -> (B, 30, 64)
        last = out[:, -1, :]         # -> (B, 64)
        return self.head(last)       # -> (B, 10)


def init_model(weights_path: str = "model_weights.pkl", device: str | None = None):
    """
    Returns a ready-to-infer CNNLSTM model with weights loaded.
    This must match the competition’s required interface.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM().to(device)
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except FileNotFoundError:
        print(f"Warning: weights file '{weights_path}' not found, returning untrained model.")
    model.eval()
    return model
