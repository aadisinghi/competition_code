# train_cnnlstm_stateful.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid forecaster (stateful-capable).
    Input:  (B, 60, 2)
    Output: (B, 10)
    """
    def __init__(self):
        super().__init__()
        # --- CNN layers ---
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2)  # 60 → 30

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

        # --- Linear head ---
        self.head = nn.Linear(64, 10)

    def forward(self, x, hc=None):
        # x: (B, 60, 2)
        x = x.transpose(1, 2)        # → (B, 2, 60)
        x = F.relu(self.conv1(x))    # → (B, 32, 60)
        x = F.relu(self.conv2(x))    # → (B, 64, 60)
        x = self.pool(x)             # → (B, 64, 30)
        x = x.transpose(1, 2)        # → (B, 30, 64)

        out, (h, c) = self.lstm(x, hc) if hc is not None else self.lstm(x)
        last = out[:, -1, :]         # → (B, 64)
        return self.head(last), (h, c)

# ==============================================================
# 2) Data Preparation
# ==============================================================
print("\n[1] Loading and preparing dataset...")
RAW_PATH = "data/New folder/train.pkl"

df = pd.read_pickle(RAW_PATH)
bad_series = [4, 23, 32, 33, 42]
df = df[~df['series_id'].isin(bad_series)].reset_index(drop=True)
df = df.sort_values(['series_id', 'time_step']).reset_index(drop=True)

unique_series = sorted(df['series_id'].unique())
n_train = int(0.8 * len(unique_series))
train_ids, val_ids = unique_series[:n_train], unique_series[n_train:]

train_df = df[df['series_id'].isin(train_ids)].reset_index(drop=True)
val_df   = df[df['series_id'].isin(val_ids)].reset_index(drop=True)

print(f"   → Train series: {len(train_ids)}, Val series: {len(val_ids)}")
print(f"   → Train rows: {len(train_df):,}, Val rows: {len(val_df):,}")

# ==============================================================
# 3) Config
# ==============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW, INPUT_LEN, HORIZON, STRIDE = 70, 60, 10, 70  # non-overlapping
EPOCHS, LR = 5, 0.001
SCALE = 1  # amplify small return targets

# ==============================================================
# 4) Window builder
# ==============================================================
def make_series_windows(df, window=70, input_len=60, horizon=10, stride=70):
    """Return dict: {series_id: [(X, y), ...]}"""
    windows = {}
    for sid, g in df.groupby("series_id"):
        g = g.sort_values("time_step").reset_index(drop=True)
        arr = g[["close", "volume"]].to_numpy(np.float32)
        n = len(arr)
        if n < window:
            continue
        seqs = []
        for s in range(0, n - window + 1, stride):
            chunk = arr[s:s + window]
            base = chunk[input_len - 1, 0]

            # normalize relative to last close in input
            x_close = chunk[:input_len, 0] / base
            x_vol   = np.log1p(np.clip(chunk[:input_len, 1], 0.0, None))
            X = np.stack([x_close, x_vol], axis=1).astype(np.float32)

            # predict scaled log returns
            y = SCALE * np.log(chunk[input_len:, 0] / base).astype(np.float32)
            seqs.append((X, y))
        windows[sid] = seqs
    return windows

print("\n[2] Building windows...")
train_windows = make_series_windows(train_df, WINDOW, INPUT_LEN, HORIZON, STRIDE)
val_windows   = make_series_windows(val_df, WINDOW, INPUT_LEN, HORIZON, STRIDE)
print(f"   → Train windows: {sum(len(v) for v in train_windows.values()):,}")
print(f"   → Val windows: {sum(len(v) for v in val_windows.values()):,}\n")

# ==============================================================
# 5) Initialize model
# ==============================================================
print("[3] Initializing CNN-LSTM model...")
model = CNNLSTM().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print("   ✓ Model ready on", DEVICE, "\n")

# ==============================================================
# 6) Training loop (per series)
# ==============================================================
print("[4] Starting training...\n")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_samples = 0.0, 0

    for sid, seqs in tqdm(train_windows.items(), desc=f"Epoch {epoch} Training", ncols=100):
        # initialize hidden and cell state for this series
        h, c = None, None

        for i, (X_np, y_np) in enumerate(seqs):
            xb = torch.from_numpy(X_np).unsqueeze(0).to(DEVICE)  # (1, 60, 2)
            yb = torch.from_numpy(y_np).unsqueeze(0).to(DEVICE)  # (1, 10)

            optimizer.zero_grad()

            # forward pass with previous hidden state
            pred, (h, c) = model(xb, (h, c) if h is not None and c is not None else None)

            # loss + backward
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # detach to truncate the computation graph but keep numeric state
            h, c = h.detach(), c.detach()

            total_loss += loss.item()
            total_samples += 1

            if i % 500 == 0 and i > 0:
                print(f"      Series {sid} → window {i}/{len(seqs)} | loss={loss.item():.6f}", flush=True)

        # reset states at the end of each series
        h, c = None, None

    train_loss = total_loss / max(total_samples, 1)
    print(f"   ✓ Epoch {epoch} complete → train_loss={train_loss:.6f}")

    # ------------------------------------------------------
    # Validation (stateful per series)
    # ------------------------------------------------------
    model.eval()
    val_loss, val_samples = 0.0, 0
    with torch.no_grad():
        for sid, seqs in val_windows.items():
            h, c = None, None
            for X_np, y_np in seqs:
                xb = torch.from_numpy(X_np).unsqueeze(0).to(DEVICE)
                yb = torch.from_numpy(y_np).unsqueeze(0).to(DEVICE)

                pred, (h, c) = model(xb, (h, c) if h is not None and c is not None else None)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                val_samples += 1

                # detach state for numeric stability
                h, c = h.detach(), c.detach()

            h, c = None, None  # reset for next series

    val_loss /= max(val_samples, 1)
    print(f"   ✓ Validation complete → val_loss={val_loss:.6f}\n")

# ==============================================================
# 7) Save weights
# ==============================================================
torch.save(model.state_dict(), "cnn_lstm_weights.pth")
print("[5] Training complete.")
print("   ✓ Saved weights → cnn_lstm_weights.pth\n")
