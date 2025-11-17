import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON = 10

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid forecaster"""
    def __init__(self, forecast_len=HORIZON):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.drop_cnn = nn.Dropout(0.15)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=200, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=200, hidden_size=200, num_layers=1, batch_first=True)
        self.drop_rnn = nn.Dropout(0.15)

        self.fc = nn.Linear(200, forecast_len)

    def forward(self, x):
        # x: (B, 60, 3)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x)); x = self.drop_cnn(x)
        x = F.relu(self.conv2(x)); x = self.drop_cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x); x = self.drop_rnn(x)
        x, _ = self.lstm2(x); x = self.drop_rnn(x)
        last = x[:, -1, :]
        return self.fc(last)

def init_model(weights_path="model_weights_new_feature.pkl", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
