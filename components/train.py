from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import math
import numpy as np
import torch

from utils.logger import Logger

# ----------------------------- Models ---------------------------------------- #
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, dim) -> (B, 1, dim)
        # x: (B, T, F)
        out1, _ = self.lstm1(x) # (B, T, 512)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1) # (B, T, 512)
        out2 = self.dropout(out2)
        last = out2[:, -1, :] # (B, 512)
        z = self.relu(self.fc2(self.relu(self.fc1(last))))
        y = self.fc_out(z).squeeze(-1)
        return y

class RNNRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=input_size, hidden_size=256, batch_first=True)
        self.rnn2 = nn.RNN(input_size=256, hidden_size=256, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, dim) -> (B, 1, dim)
        # x: (B, T, F)
        out1, _ = self.rnn1(x) # (B, T, 512)
        out1 = self.dropout(out1)
        out2, _ = self.rnn2(out1) # (B, T, 512)
        out2 = self.dropout(out2)
        last = out2[:, -1, :] # (B, 512)
        z = self.relu(self.fc1(last))
        y = self.fc_out(z).squeeze(-1)
        return y

class GRURegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=256, batch_first=True)
        self.gru2 = nn.GRU(input_size=256, hidden_size=256, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, dim) -> (B, 1, dim)
        # x: (B, T, F)
        out1, _ = self.gru1(x) # (B, T, 512)
        out1 = self.dropout(out1)
        out2, _ = self.gru2(out1) # (B, T, 512)
        out2 = self.dropout(out2)
        last = out2[:, -1, :] # (B, 512)
        z = self.relu(self.fc2(self.relu(self.fc1(last))))
        y = self.fc_out(z).squeeze(-1)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(25600.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        d_model = 64
        self.embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(d_model, 25)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(25, 1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, dim) -> (B, 1, dim)
        # x: (B, T, F)
        x = self.embed(x) # (B, T, 64)
        x = self.pos_encoder(x)
        out = self.transformer(x) # (B, T, 64)
        last = out[:, -1, :] # (B, 64)
        z = self.relu(self.fc1(last))
        y = self.fc_out(z).squeeze(-1)
        return y

class TabMLPRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)
    def forward(self, x):
        # x: (B, flat_dim)
        x = x.view(x.size(0), -1)
        z = self.relu(self.fc1(x))
        z = self.dropout(z)
        z = self.relu(self.fc2(z))
        z = self.dropout(z)
        z = self.relu(self.fc3(z))
        y = self.fc_out(z).squeeze(-1)
        return y

# ----------------------------- Metrics -------------------------------------- #
def mse(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))

# ----------------------------- Training ------------------------------------- #
def _loss_stats(losses: List[float]) -> Dict[str, float]:
    if not losses:
        return {k: float("nan") for k in ["mean", "median", "std", "min", "max"]}
    a = np.asarray(losses, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }

def train_model(
    dl_train,
    dl_val,
    input_size: int,
    original_y_train: np.ndarray,  # Kept for compatibility/verification, but not used for metrics
    original_y_val: np.ndarray,  # Kept for compatibility/verification, but not used for metrics
    model_type: str,
    weight_decay: float,
    dropout_rate: float,
    epochs: int,
    patience: int,
    lr: float,
    save_dir: Optional[Path] = None,
    ticker: Optional[str] = None,
    load_path: Optional[Path] = None,
    log_per_batch_debug: bool = False,
    y_scaler: Optional[MinMaxScaler] = None, # <-- NEW: needed to compute price-scale metrics
):
    """
    Trains and saves best validation checkpoint to `save_dir` with epoch in filename.
    Optimizes MSE on the scaled target, but logs *also* the metrics on the original price scale
    (via `y_scaler`). Early stopping + best checkpoint are chosen by *price-scale* val MSE.
    """
    if y_scaler is None:
        raise ValueError("train_model: y_scaler must be provided to compute price-scale metrics.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "lstm":
        model = LSTMRegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    elif model_type == "rnn":
        model = RNNRegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    elif model_type == "gru":
        model = GRURegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    elif model_type == "transformer":
        model = TransformerRegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    elif model_type == "tabmlp":
        model = TabMLPRegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    # Optionally load initial weights
    if load_path is not None:
        Logger.info(f"Loading initial weights from: {load_path}")
        state = torch.load(load_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    # Trackers
    best_state = None
    best_epoch = -1
    best_val_price_mse = float("inf")
    best_train_price_mse = float("inf")
    best_val_scaled_mse = float("inf")
    best_train_scaled_mse = float("inf")
    best_val_price_mae = float("inf")
    best_val_price_rmse = float("inf")
    best_train_price_mae = float("inf")
    best_train_price_rmse = float("inf")
    bad_epochs = 0
    train_hist_scaled, val_hist_scaled = [], []
    train_hist_price_mse, val_hist_price_mse = [], []
    train_hist_price_mae, val_hist_price_mae = [], []
    train_hist_price_rmse, val_hist_price_rmse = [], []
    save_path_for_best = ""
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tr_losses = []
        tr_pred_s_batches = []
        tr_true_s_batches = []  # ADDED
        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb = xb.to(device) # (B, T, F)
            yb = yb.to(device) # (B,)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb) # scaled loss
            loss.backward()
            opt.step()
            l = float(loss.item())
            tr_losses.append(l)
            # accumulate for price-scale metrics
            tr_pred_s_batches.append(pred.detach().cpu().numpy().ravel())
            tr_true_s_batches.append(yb.cpu().numpy().ravel())
            if log_per_batch_debug and step % 50 == 0:
                Logger.debug(f"[train] epoch {ep} step {step} loss_scaled={l:.6f}")
        # scaled metrics (epoch-level)
        tr_stats = _loss_stats(tr_losses)
        if tr_pred_s_batches:
            y_pred_s = np.concatenate(tr_pred_s_batches)
            y_true_s = np.concatenate(tr_true_s_batches)
            tr_mse_scaled = mse(y_true_s, y_pred_s)
            tr_mae_scaled = mae(y_true_s, y_pred_s)
            tr_rmse_scaled = rmse(y_true_s, y_pred_s)
            y_pred_price = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_true_price = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
            tr_mse_price = mse(y_true_price, y_pred_price)
            tr_mae_price = mae(y_true_price, y_pred_price)
            tr_rmse_price = rmse(y_true_price, y_pred_price)
        else:
            tr_mse_scaled = tr_mae_scaled = tr_rmse_scaled = float("nan")
            tr_mse_price = tr_mae_price = tr_rmse_price = float("nan")
        # ---- Val ----
        model.eval()
        val_losses = []
        val_pred_s_batches = []
        val_true_s_batches = []
        with torch.no_grad():
            for step, (xb, yb) in enumerate(dl_val, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb) # scaled
                l = float(loss.item())
                val_losses.append(l)
                val_pred_s_batches.append(pred.detach().cpu().numpy().ravel())
                val_true_s_batches.append(yb.cpu().numpy().ravel())
                if log_per_batch_debug and step % 50 == 0:
                    Logger.debug(f"[val] epoch {ep} step {step} loss_scaled={l:.6f}")
        val_stats = _loss_stats(val_losses)
        if val_pred_s_batches:
            y_pred_s = np.concatenate(val_pred_s_batches)
            y_true_s = np.concatenate(val_true_s_batches)
            val_mse_scaled = mse(y_true_s, y_pred_s)
            val_mae_scaled = mae(y_true_s, y_pred_s)
            val_rmse_scaled = rmse(y_true_s, y_pred_s)
            y_pred_price = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_true_price = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
            val_mse_price = mse(y_true_price, y_pred_price)
            val_mae_price = mae(y_true_price, y_pred_price)
            val_rmse_price = rmse(y_true_price, y_pred_price)
        else:
            val_mse_scaled = val_mae_scaled = val_rmse_scaled = float("nan")
            val_mse_price = val_mae_price = val_rmse_price = float("nan")
        # histories
        train_hist_scaled.append(tr_mse_scaled)
        val_hist_scaled.append(val_mse_scaled)
        train_hist_price_mse.append(tr_mse_price)
        val_hist_price_mse.append(val_mse_price)
        train_hist_price_mae.append(tr_mae_price)
        val_hist_price_mae.append(val_mae_price)
        train_hist_price_rmse.append(tr_rmse_price)
        val_hist_price_rmse.append(val_rmse_price)
        Logger.info(
            "Epoch %03d | "
            "train (scaled) MSE: mean=%.6f med=%.6f std=%.6f min=%.6f max=%.6f | "
            "val (scaled) MSE: mean=%.6f med=%.6f std=%.6f min=%.6f max=%.6f",
            ep, tr_stats["mean"], tr_stats["median"], tr_stats["std"], tr_stats["min"], tr_stats["max"],
            val_stats["mean"], val_stats["median"], val_stats["std"], val_stats["min"], val_stats["max"],
        )
        Logger.info(
            "Epoch %03d | train (price) MSE=%.6f MAE=%.6f RMSE=%.6f | "
            "val (price) MSE=%.6f MAE=%.6f RMSE=%.6f",
            ep, tr_mse_price, tr_mae_price, tr_rmse_price, val_mse_price, val_mae_price, val_rmse_price
        )
        # Early stopping + save best *by price-scale val MSE*
        if val_mse_price < (best_val_price_mse - 1e-12):
            best_val_price_mse = val_mse_price
            best_epoch = ep
            best_train_price_mse = tr_mse_price
            best_val_scaled_mse = val_mse_scaled
            best_train_scaled_mse = tr_mse_scaled
            best_val_price_mae = val_mae_price
            best_val_price_rmse = val_rmse_price
            best_train_price_mae = tr_mae_price
            best_train_price_rmse = tr_rmse_price
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_dir is not None:
                prefix = f"{ticker}_" if ticker else ""
                save_path = save_dir / f"{prefix}best-epoch{ep:03d}.pt"
                torch.save(best_state, save_path)
                save_path_for_best = str(save_path.resolve())
                Logger.info(
                    f"New best (price) val MSE={best_val_price_mse:.6f} at epoch {ep}; "
                    f"saved → {save_path_for_best}"
                )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                Logger.info(
                    f"Early stopping at epoch {ep} "
                    f"(best price val MSE={best_val_price_mse:.6f} @ epoch {best_epoch})"
                )
                break
    # Load best state at end
    if best_state is not None:
        model.load_state_dict(best_state)
    out_info = {
        "best_epoch": best_epoch,
        # price-scale (canonical)
        "best_val_mse": float(best_val_price_mse),
        "best_train_mse": float(best_train_price_mse),
        "best_val_mae": float(best_val_price_mae),
        "best_val_rmse": float(best_val_price_rmse),
        "best_train_mae": float(best_train_price_mae),
        "best_train_rmse": float(best_train_price_rmse),
        # scaled (for reference)
        "best_val_mse_scaled": float(best_val_scaled_mse),
        "best_train_mse_scaled": float(best_train_scaled_mse),
        "best_path": save_path_for_best,
        "epochs_run": len(train_hist_scaled),
        # finals
        "final_train_mse": float(train_hist_price_mse[-1]) if train_hist_price_mse else float("nan"),
        "final_val_mse": float(val_hist_price_mse[-1]) if val_hist_price_mse else float("nan"),
        "final_train_mse_scaled": float(train_hist_scaled[-1]) if train_hist_scaled else float("nan"),
        "final_val_mse_scaled": float(val_hist_scaled[-1]) if val_hist_scaled else float("nan"),
    }
    price_hist = {
        "train_mse": train_hist_price_mse,
        "val_mse": val_hist_price_mse,
        "train_mae": train_hist_price_mae,
        "val_mae": val_hist_price_mae,
        "train_rmse": train_hist_price_rmse,
        "val_rmse": val_hist_price_rmse,
    }
    return model, train_hist_scaled, val_hist_scaled, out_info, price_hist

# ----------------------------- Evaluation ----------------------------------- #
def evaluate(model: nn.Module, dl_test: DataLoader, y_scaler: MinMaxScaler, original_y_test: np.ndarray):  # original_y_test kept for compatibility, but not used
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().ravel()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy().ravel())
    if not preds_scaled:
        return {
            "test_MSE": float("nan"),
            "test_MAE": float("nan"),
            "test_RMSE": float("nan"),
            "test_MSE_scaled": float("nan"),
            "test_MAE_scaled": float("nan"),
            "test_RMSE_scaled": float("nan"),
            "y_true": np.array([]),
            "y_pred": np.array([]),
        }
    y_pred_s = np.concatenate(preds_scaled)
    y_true_s = np.concatenate(trues_scaled)
    # scaled metrics (0..1)
    scaled_MSE = mse(y_true_s, y_pred_s)
    scaled_MAE = mae(y_true_s, y_pred_s)
    scaled_RMSE = rmse(y_true_s, y_pred_s)
    # Inverse to intermediate scale (residual or direct)
    y_pred_inter = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    # Final to original target scale
    y_pred = y_pred_inter
    y_true = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    return {
        "test_MSE": mse(y_true, y_pred),
        "test_MAE": mae(y_true, y_pred),
        "test_RMSE": rmse(y_true, y_pred),
        "test_MSE_scaled": scaled_MSE,
        "test_MAE_scaled": scaled_MAE,
        "test_RMSE_scaled": scaled_RMSE,
        "y_true": y_true,
        "y_pred": y_pred,
    }