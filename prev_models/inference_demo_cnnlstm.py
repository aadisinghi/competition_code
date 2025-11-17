# inference.py
import torch
import pandas as pd
import numpy as np
import time
from submission.prev_models.model_demo_cnnlstm import init_model

def generate_forecast(x_test_path: str) -> pd.DataFrame:
    """
    Generates submission.pkl following competition format.
    Input:
        x_test_path: path to x_test.pkl (must contain window_id, time_step, close, volume)
    Output:
        submission.pkl saved to current directory.
    """
    # --------------------------
    # 1) Load model and device
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model("model_weights.pkl", device=device)
    model.eval()

    # --------------------------
    # 2) Load test data
    # --------------------------
    print("Loading x_test data...")
    t0 = time.perf_counter()
    xt = pd.read_pickle(x_test_path).sort_values(["window_id", "time_step"]).reset_index(drop=True)
    print(f"Loaded {len(xt)} rows in {time.perf_counter() - t0:.2f}s")

    # --------------------------
    # 3) Inference
    # --------------------------
    print("Running inference...\n.")
    rows = []
    EPS = 1e-8

    with torch.no_grad():
        for wid, g in xt.groupby("window_id", sort=True):
            g = g.sort_values("time_step")
            x_close = g["close"].values
            x_vol   = g["volume"].values

            # ensure window length = 60
            if len(x_close) != 60:
                continue

            # --- identical normalization as training ---
            close = np.log(np.clip(x_close, EPS, None))
            vol   = np.log1p(np.clip(x_vol, 0.0, None))
            X = np.stack([close, vol], axis=1).astype(np.float32)  # (60, 2)
            xb = torch.from_numpy(X).unsqueeze(0).to(device)       # (1, 60, 2)

            # --- model prediction (log-space) ---
            pred_log = model(xb).cpu().numpy().ravel()             # (10,)

            # convert back to actual price scale
            pred_close = np.exp(pred_log)

            for h, yhat in enumerate(pred_close):
                rows.append((wid, h, float(yhat)))

    # --------------------------
    # 4) Build and save submission
    # --------------------------
    submission = pd.DataFrame(rows, columns=["window_id", "time_step", "pred_close"])
    submission = submission.sort_values(["window_id", "time_step"]).reset_index(drop=True)
    submission.to_pickle("submission.pkl")

    print(f"Saved submission.pkl with {len(submission)} rows.")
    return submission


if __name__ == "__main__":
    # Example usage:
    generate_forecast("x_test.pkl")
