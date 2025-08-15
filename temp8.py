from __future__ import annotations
# ---------------- Runtime stamp for output names ---------------- #
from datetime import datetime
runtime = datetime.now().strftime("%Y%m%d-%H%M%S")
import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import pytz
from dataclasses import dataclass
from datetime import time, date
from dateutil import parser as dtparser
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Iterable, List, Optional, Tuple, Dict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from statsmodels.tsa.arima.model import ARIMA
# ----------------------------- Config & Utilities ----------------------------- #
@dataclass(frozen=True)
class Paths:
    fnspid_news_csv: str
    kaggle_news_csv: str
    prices_csv: str
    out_sentiment_csv: str
@dataclass(frozen=True)
class Schema:
    news_ticker: str
    news_time: str
    news_title: Optional[str]
    news_body: Optional[str]
    price_ticker: Optional[str]
    price_date: str
    price_open: Optional[str]
    price_high: Optional[str]
    price_low: Optional[str]
    price_close: str
    price_volume: Optional[str]
@dataclass
class Settings: # Non-frozen for dynamic updates
    market_tz: str = "America/New_York"
    market_close_str: str = "16:00"
    batch_size: int = 2 ** 13
    max_length: int = 2 ** 13 # tokenizer truncation
    text_joiner: str = " "
    lower_case_tickers: bool = False
    assume_news_timezone: str = "UTC"
    dedupe_titles: bool = True
    titles_only: bool = True # paper used titles; set False to include body too
def _coerce_datetime(x: object, assume_tz: pytz.BaseTzInfo) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        ts = x
    else:
        try:
            ts = pd.Timestamp(dtparser.parse(str(x)))
        except Exception:
            return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize(assume_tz).tz_convert(pytz.UTC)
    return ts.tz_convert(pytz.UTC)
def _ensure_utc(series: pd.Series, assume_tz_name: str) -> pd.Series:
    tz = pytz.timezone(assume_tz_name)
    return series.map(lambda x: _coerce_datetime(x, tz))
def _normalize_ticker(s: pd.Series, lower_case: bool) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.str.lower() if lower_case else s.str.upper()
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
# ----------------------------- Trading Calendar ------------------------------ #
class TradingCalendar:
    """
    Trading calendar derived from the prices table.
    Provides next_trading_day(date) by binary search over sorted unique dates.
    """
    def __init__(self, trading_days: List[date]):
        if not trading_days:
            raise ValueError("Empty trading_days.")
        self._days = sorted(set(trading_days))
    @property
    def days(self) -> List[date]:
        return self._days
    def next_trading_day(self, d: date) -> date:
        idx = _bisect_left(self._days, d)
        if idx == len(self._days):
            raise ValueError(f"No trading day on/after {d} in calendar.")
        return self._days[idx]
def _bisect_left(a: List[date], x: date) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
def build_calendar(prices_df: pd.DataFrame, schema: Schema) -> TradingCalendar:
    dates = pd.to_datetime(prices_df[schema.price_date], errors="coerce").dt.date
    unique_days = dates.dropna().unique().tolist()
    return TradingCalendar(unique_days)
# ----------------------------- FinBERT Scoring ------------------------------- #
class FinBertScorer:
    """
    ProsusAI/finbert off-the-shelf; returns p_pos, p_neg, p_neu, sentiment_score=p_pos-p_neg
    """
    def __init__(self, batch_size: int, max_length: int, model=None):
        if model is None:
            model_id = "ProsusAI/finbert"
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                "text-classification",
                model=AutoModelForSequenceClassification.from_pretrained(model_id),
                tokenizer=AutoTokenizer.from_pretrained(model_id),
                device=device,
                return_all_scores=True,
                truncation=True
            )
        else:
            self.pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=model.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True,
                truncation=True
            )
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            self.pipe.model = torch.compile(self.pipe.model) # Compile the model
        self.batch_size = batch_size
        self.max_length = max_length
    def score_texts(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["p_pos", "p_neg", "p_neu", "sentiment_score"])
        results = []
        self.pipe.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT", unit="batch"):
                batch = texts[i:i + self.batch_size]
                inputs = self.pipe.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.pipe.device) for k, v in inputs.items()}
                with autocast(): # Enable mixed precision
                    outputs = self.pipe.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for prob in probs:
                    p_pos, p_neg, p_neu = prob
                    results.append((p_pos, p_neg, p_neu, p_pos - p_neg))
        return pd.DataFrame(results, columns=["p_pos", "p_neg", "p_neu", "sentiment_score"])
# ----------------------------- Fine-Tuning FinBERT with NSI ------------------ #
class NSIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
def compute_nsi(prices_df: pd.DataFrame, schema: Schema, threshold: float = 0.01) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["trading_date"] = pd.to_datetime(prices[schema.price_date]).dt.date
    prices["return"] = (prices[schema.price_close] - prices[schema.price_open]) / prices[schema.price_open]
    prices["NSI"] = np.where(prices["return"] > threshold, 0,
                             np.where(prices["return"] < -threshold, 1, 2)) # Map: pos=0, neg=1, neu=2 to match FinBERT labels
    return prices[["trading_date", "NSI"]]
def fine_tune_finbert(news_df: pd.DataFrame, prices_df: pd.DataFrame, schema: Schema, s: Settings, prep: NewsPreprocessor):
    # Compute NSI per day
    nsi_df = compute_nsi(prices_df, schema)
    # Map news to trading_date and label with NSI
    news_df["trading_date"] = prep.compute_trading_date(news_df["published_utc"]).dt.date
    labeled_df = pd.merge(news_df, nsi_df, on="trading_date", how="inner")
    labeled_df = labeled_df.dropna(subset=["NSI"])
    texts = prep.build_text(labeled_df).tolist()
    labels = labeled_df["NSI"].astype(int).tolist() # 0: pos, 1: neg, 2: neu
    # Load base FinBERT
    model_id = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
    # Dataset
    train_dataset = NSIDataset(texts, labels, tokenizer, s.max_length)
    # Trainer
    training_args = TrainingArguments(
        output_dir="./finbert_finetuned",
        num_train_epochs=3, # Paper suggests task-specific fine-tuning; adjust as needed
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    # Save and return fine-tuned model
    model.save_pretrained("./finbert_finetuned")
    tokenizer.save_pretrained("./finbert_finetuned")
    return AutoModelForSequenceClassification.from_pretrained("./finbert_finetuned"), tokenizer
# ----------------------------- Pipeline Stages ------------------------------- #
class NewsPreprocessor:
    def __init__(self, settings: Settings, schema: Schema, calendar: TradingCalendar):
        self.s = settings
        self.schema = schema
        self.calendar = calendar
        self.market_tz = pytz.timezone(settings.market_tz)
        hh, mm = map(int, settings.market_close_str.split(":"))
        self.market_close = time(hour=hh, minute=mm)
    def build_text(self, df: pd.DataFrame) -> pd.Series:
        title = df[self.schema.news_title].fillna("").astype(str)
        if self.s.titles_only or self.schema.news_body is None:
            text = title
        else:
            body = df[self.schema.news_body].fillna("").astype(str)
            text = (title + self.s.text_joiner + body).str.strip()
        return text.str.replace(r"\s+", " ", regex=True)
    def normalize_time_columns(self, df: pd.DataFrame) -> pd.Series:
        return _ensure_utc(df[self.schema.news_time], self.s.assume_news_timezone)
    def compute_trading_date(self, published_utc: pd.Series) -> pd.Series:
        local_ts = published_utc.dt.tz_convert(self.market_tz)
        after_close = local_ts.dt.time > self.market_close
        candidate_date = local_ts.dt.date.where(~after_close, (local_ts + pd.Timedelta(days=1)).dt.date)
        mapped = candidate_date.map(self._map_to_trading_day)
        return pd.to_datetime(mapped)
    def _map_to_trading_day(self, d: date) -> date:
        return self.calendar.next_trading_day(d)
    def normalize_tickers(self, df: pd.DataFrame) -> pd.Series:
        return _normalize_ticker(df[self.schema.news_ticker], self.s.lower_case_tickers)
# ----------------------------- Dataset / Dataloader -------------------------- #
class SequenceDataset(Dataset):
    """
    Produces sequences of length lookback with features at each step,
    target is next-day Close (scaled separately).
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, lookback: int):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.lb = lookback
        self.n = len(df)
    def __len__(self):
        return max(0, self.n - self.lb)
    def __getitem__(self, idx):
        x = self.X[idx: idx + self.lb] # (lookback, F)
        y = self.y[idx + self.lb] # scalar target at t+1
        return torch.from_numpy(x), torch.tensor(y)
# ----------------------------- Model ---------------------------------------- #
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(100, 25)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(25, 1)
    def forward(self, x):
        # x: (B, T, F)
        out1, _ = self.lstm1(x) # (B, T, 100)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1) # (B, T, 100)
        out2 = self.dropout(out2)
        last = out2[:, -1, :] # (B, 100)
        z = self.relu(self.fc1(last))
        y = self.fc_out(z).squeeze(-1)
        return y
# ----------------------------- Metrics -------------------------------------- #
def mse(y_true, y_pred): return float(np.mean((y_true - y_pred) ** 2))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))
# ----------------------------- Features & Scaling --------------------------- #
def prepare_joined_frame(sent_daily: pd.DataFrame, prices_df: pd.DataFrame, schema: Schema, ticker: str) -> pd.DataFrame:
    prices = prices_df.copy()
    if schema.price_ticker:
        prices["ticker"] = _normalize_ticker(prices[schema.price_ticker], False)
    else:
        prices["ticker"] = ticker
    prices["trading_date"] = pd.to_datetime(prices[schema.price_date], errors="coerce").dt.date
    keep = ["ticker", "trading_date", schema.price_close]
    for c in [schema.price_open, schema.price_high, schema.price_low, schema.price_volume]:
        if c: keep.append(c)
    prices = prices[keep].dropna(subset=["ticker", "trading_date"])
    colmap = {schema.price_close: "Close"}
    if schema.price_open: colmap[schema.price_open] = "Open"
    if schema.price_high: colmap[schema.price_high] = "High"
    if schema.price_low: colmap[schema.price_low] = "Low"
    if schema.price_volume: colmap[schema.price_volume] = "Volume"
    prices = prices.rename(columns=colmap)
    joined = pd.merge(prices, sent_daily, on=["ticker", "trading_date"], how="left").sort_values(
        ["ticker", "trading_date"]
    )
    joined["SentimentScore"] = joined["SentimentScore"].fillna(0.0)
    joined["N_t"] = joined["N_t"].fillna(0)
    return joined.reset_index(drop=True)
def build_supervised_for_ticker(df_all: pd.DataFrame, ticker: str, lookback: int, predict_returns: bool) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble features (no scaling here) and keep raw Close for later target scaling.
    """
    df = df_all[df_all["ticker"] == ticker].copy().sort_values("trading_date").reset_index(drop=True)
    feat_cols = ["SentimentScore"]
    for c in ["Open", "High", "Low", "Volume", "Close"]:
        if c in df.columns:
            feat_cols.append(c)
    if predict_returns:
        df['raw_target'] = (df['Close'] - df['Open']) / df['Open']
    else:
        df['raw_target'] = df['Close']
    return df, feat_cols
def temporal_split(df: pd.DataFrame, train_ratio: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(math.floor(n * train_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
def split_train_val(train_df: pd.DataFrame, val_ratio_within_train: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    cut = int(math.floor(n * (1 - val_ratio_within_train)))
    return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()
def make_dataloaders_for_ticker(df: pd.DataFrame, feat_cols: List[str], lookback: int, use_arima: bool, batch_size: int = 32):
    """
    Chronological split; fit scalers on TRAIN only; transform all splits; build PyTorch loaders.
    """
    # Splits
    train_all, test = temporal_split(df, train_ratio=0.9)
    train, val = split_train_val(train_all, val_ratio_within_train=0.1)
    # Fit X scaler on TRAIN only
    x_scaler = MinMaxScaler()
    x_scaler.fit(train[feat_cols].astype(float))
    # Transform X across splits
    for part in [train, val, test]:
        part.loc[:, feat_cols] = x_scaler.transform(part[feat_cols].astype(float))
    # Hybrid ARIMA if enabled
    is_hybrid = use_arima
    arima_pred_train = arima_pred_val = arima_pred_test = None
    if is_hybrid:
        arima_order = (1, 1, 2)
        # Fit on train for train fitted and val forecast
        arima_fit = ARIMA(train['raw_target'], order=arima_order).fit()
        arima_pred_train = arima_fit.fittedvalues.values
        arima_pred_val = arima_fit.forecast(steps=len(val)).values
        # Fit on train_all for test forecast
        arima_fit_full = ARIMA(train_all['raw_target'], order=arima_order).fit()
        arima_pred_test = arima_fit_full.forecast(steps=len(test)).values
        # Set y to residuals
        train['y'] = train['raw_target'] - arima_pred_train
        val['y'] = val['raw_target'] - arima_pred_val
        test['y'] = test['raw_target'] - arima_pred_test
    else:
        train['y'] = train['raw_target']
        val['y'] = val['raw_target']
        test['y'] = test['raw_target']
    # Fit y scaler on TRAIN only
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[['y']])
    # Transform y (residuals or raw target)
    train['y'] = y_scaler.transform(train[['y']]).ravel()
    val['y'] = y_scaler.transform(val[['y']]).ravel()
    test['y'] = y_scaler.transform(test[['y']]).ravel()
    # Datasets
    ds_train = SequenceDataset(train, feat_cols, 'y', lookback)
    ds_val = SequenceDataset(val, feat_cols, 'y', lookback)
    ds_test = SequenceDataset(test, feat_cols, 'y', lookback)
    # Loaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    # For metrics on original target scale
    original_y_train = train['raw_target'].iloc[lookback:].values
    original_y_val = val['raw_target'].iloc[lookback:].values
    original_y_test = test['raw_target'].iloc[lookback:].values
    if is_hybrid:
        arima_pred_train_slice = arima_pred_train[lookback:]
        arima_pred_val_slice = arima_pred_val[lookback:]
        arima_pred_test_slice = arima_pred_test[lookback:]
    else:
        arima_pred_train_slice = np.zeros_like(original_y_train)
        arima_pred_val_slice = np.zeros_like(original_y_val)
        arima_pred_test_slice = np.zeros_like(original_y_test)
    return (
        dl_train, dl_val, dl_test, y_scaler, is_hybrid,
        arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice,
        original_y_train, original_y_val, original_y_test
    )
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
def _find_weight_file_for_ticker(load_dir: Path, ticker: str) -> Optional[Path]:
    if load_dir.is_file():
        return load_dir
    if not load_dir.exists() or not load_dir.is_dir():
        return None
    # Prefer files that contain the ticker symbol
    cand = sorted(list(load_dir.glob(f"*{ticker}*.pt")) + list(load_dir.glob(f"*{ticker}*.pth")))
    if cand:
        return cand[0]
    # Fallback: any .pt/.pth
    any_cand = sorted(list(load_dir.glob("*.pt")) + list(load_dir.glob("*.pth")))
    return any_cand[0] if any_cand else None
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
    is_hybrid: bool,
    arima_pred_train_slice: np.ndarray,
    arima_pred_val_slice: np.ndarray,
    original_y_train: np.ndarray,
    original_y_val: np.ndarray,
    epochs: int = 100,
    patience: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    dropout_rate: float = 0.2,
    device: str = None,
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_size=input_size, dropout_rate=dropout_rate).to(device)
    # Optionally load initial weights
    if load_path is not None and Path(load_path).exists():
        logging.info(f"Loading initial weights from: {load_path}")
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
        tr_pred_s_batches, tr_true_s_batches = [], []
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
            tr_true_s_batches.append(yb.detach().cpu().numpy().ravel())
            if log_per_batch_debug and step % 50 == 0:
                logging.debug(f"[train] epoch {ep} step {step} loss_scaled={l:.6f}")
        # scaled metrics (epoch-level)
        tr_stats = _loss_stats(tr_losses)
        tr_mse_scaled = tr_stats["mean"]
        # price-scale metrics (epoch-level)
        if tr_pred_s_batches:
            y_pred_s = np.concatenate(tr_pred_s_batches)
            y_true_s = np.concatenate(tr_true_s_batches)
            y_pred_inter = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_true_inter = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
            y_pred_price = arima_pred_train_slice + y_pred_inter
            y_true_price = original_y_train
            tr_mse_price = mse(y_true_price, y_pred_price)
            tr_mae_price = mae(y_true_price, y_pred_price)
            tr_rmse_price = rmse(y_true_price, y_pred_price)
        else:
            tr_mse_price = tr_mae_price = tr_rmse_price = float("nan")
        # ---- Val ----
        model.eval()
        val_losses = []
        val_pred_s_batches, val_true_s_batches = [], []
        with torch.no_grad():
            for step, (xb, yb) in enumerate(dl_val, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb) # scaled
                l = float(loss.item())
                val_losses.append(l)
                val_pred_s_batches.append(pred.detach().cpu().numpy().ravel())
                val_true_s_batches.append(yb.detach().cpu().numpy().ravel())
                if log_per_batch_debug and step % 50 == 0:
                    logging.debug(f"[val] epoch {ep} step {step} loss_scaled={l:.6f}")
        val_stats = _loss_stats(val_losses)
        val_mse_scaled = val_stats["mean"]
        if val_pred_s_batches:
            y_pred_s = np.concatenate(val_pred_s_batches)
            y_true_s = np.concatenate(val_true_s_batches)
            y_pred_inter = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            y_true_inter = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
            y_pred_price = arima_pred_val_slice + y_pred_inter
            y_true_price = original_y_val
            val_mse_price = mse(y_true_price, y_pred_price)
            val_mae_price = mae(y_true_price, y_pred_price)
            val_rmse_price = rmse(y_true_price, y_pred_price)
        else:
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
        # logging: both scales
        logging.info(
            "Epoch %03d | "
            "train (scaled) MSE: mean=%.6f med=%.6f std=%.6f min=%.6f max=%.6f | "
            "val (scaled) MSE: mean=%.6f med=%.6f std=%.6f min=%.6f max=%.6f",
            ep, tr_stats["mean"], tr_stats["median"], tr_stats["std"], tr_stats["min"], tr_stats["max"],
            val_stats["mean"], val_stats["median"], val_stats["std"], val_stats["min"], val_stats["max"],
        )
        logging.info(
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
                logging.info(
                    f"New best (price) val MSE={best_val_price_mse:.6f} at epoch {ep}; "
                    f"saved → {save_path_for_best}"
                )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(
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
def evaluate(model: nn.Module, dl_test: DataLoader, y_scaler: MinMaxScaler, is_hybrid: bool, arima_pred_test_slice: np.ndarray, original_y_test: np.ndarray, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    y_true_inter = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    # Final to original target scale
    y_pred = arima_pred_test_slice + y_pred_inter
    y_true = original_y_test
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
# ----------------------------- News IO -------------------------------------- #
def load_fnspid_news(fnspid_news: str, use_bodies: bool) -> pd.DataFrame:
    usecols = ["Date", "Article_title", "Stock_symbol"]
    rename = {"Article_title": "title", "Stock_symbol": "stock", "Date": "date"}
    if use_bodies:
        usecols.append("Article_content")
        rename["Article_content"] = "body"
    dtypes = {
        "Article_title": str,
        "Stock_symbol": str,
        "Article_content": str if use_bodies else None
    }
    news_df = pd.read_csv(
        fnspid_news,
        usecols=usecols,
        dtype={k: v for k, v in dtypes.items() if v is not None},
        parse_dates=["Date"],
        date_format="%Y-%m-%d"
    )
    news_df = news_df.rename(columns=rename)
    return news_df
def load_kaggle_news(kaggle_news: str, use_bodies: bool) -> pd.DataFrame:
    usecols = ["title", "date", "stock"]
    if use_bodies:
        usecols.append("text") # Assuming 'text' is body column in primary dataset
    news_df = pd.read_csv(
        kaggle_news,
        usecols=usecols,
        parse_dates=["date"]
    )
    if use_bodies and "text" in news_df.columns:
        news_df = news_df.rename(columns={"text": "body"})
    elif use_bodies:
        logging.warning("No body column found in Kaggle CSV; falling back to titles only.")
    return news_df
def load_and_filter_news(fnspid_news: str, kaggle_news: str, data_source: str, start_date: pd.Timestamp, end_date: pd.Timestamp, use_bodies: bool, x_csv: Optional[str] = None) -> pd.DataFrame:
    if data_source == "fnspid":
        news_df = load_fnspid_news(fnspid_news, use_bodies)
    elif data_source == "kaggle":
        news_df = load_kaggle_news(kaggle_news, use_bodies)
    elif data_source == "both":
        fnspid_df = load_fnspid_news(fnspid_news, use_bodies)
        kaggle_df = load_kaggle_news(kaggle_news, use_bodies)
        news_df = pd.concat([fnspid_df, kaggle_df], ignore_index=True).drop_duplicates(subset=["title"])
    else:
        raise ValueError(f"Invalid data_source: {data_source}. Choose 'fnspid', 'kaggle', or 'both'.")
    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_df = news_df[(news_df["date"] > start_date) & (news_df["date"] < end_date)].dropna(subset=["date"])
    if x_csv:
        x_df = pd.read_csv(x_csv, parse_dates=["date"])
        x_df = x_df.rename(columns={"text": "title"})  # Treat X posts as titles
        x_df['body'] = None
        news_df = pd.concat([news_df, x_df], ignore_index=True).drop_duplicates(subset=["title"])
    return news_df
def make_daily_sentiment(news_df: pd.DataFrame, prices_df: pd.DataFrame,
                         paths: Paths, s: Settings, schema: Schema, fine_tune: bool) -> pd.DataFrame:
    calendar = build_calendar(prices_df, schema)
    prep = NewsPreprocessor(s, schema, calendar)
    ticker_col = schema.news_ticker
    time_col = schema.news_time
    title_col = schema.news_title
    news_df[ticker_col] = _normalize_ticker(news_df[ticker_col], s.lower_case_tickers)
    news_df["published_utc"] = _ensure_utc(news_df[time_col], s.assume_news_timezone)
    if s.dedupe_titles:
        news_df = news_df.drop_duplicates(subset=[title_col]).reset_index(drop=True)
    news_df["text"] = prep.build_text(news_df)
    news_df = news_df.dropna(subset=[ticker_col, "published_utc"]).reset_index(drop=True)
    news_df = news_df[news_df["text"].str.len() > 0].reset_index(drop=True)
    model = None
    if fine_tune:
        model, tokenizer = fine_tune_finbert(news_df, prices_df, schema, s, prep)
        scorer = FinBertScorer(s.batch_size, s.max_length, model=model)
    else:
        scorer = FinBertScorer(s.batch_size, s.max_length)
    score_df = scorer.score_texts(news_df["text"].tolist())
    assert len(score_df) == len(news_df), "Score length mismatch."
    news_df = pd.concat([news_df, score_df], axis=1)
    news_df["trading_date"] = prep.compute_trading_date(news_df["published_utc"])
    grp = news_df.groupby([ticker_col, "trading_date"], as_index=False).agg(
        N_t=("sentiment_score", "size"),
        SentimentScore=("sentiment_score", "mean"),
        p_pos_mean=("p_pos", "mean"),
        p_neg_mean=("p_neg", "mean"),
        p_neu_mean=("p_neu", "mean"),
    )
    grp = grp.rename(columns={ticker_col: "ticker"})
    grp["trading_date"] = pd.to_datetime(grp["trading_date"]).dt.date
    os.makedirs(os.path.dirname(paths.out_sentiment_csv) or ".", exist_ok=True)
    grp.sort_values(["ticker", "trading_date"]).to_csv(paths.out_sentiment_csv, index=False)
    logging.info(f"Wrote daily sentiment → {paths.out_sentiment_csv}")
    return grp
# ----------------------------------- Main ------------------------------------ #
def _list_all_tickers(prices_dir: Path) -> List[str]:
    return sorted({f.stem.upper() for f in prices_dir.glob("*.csv")})
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnspid-news", default="~/Projects/finBert/FNSPID/Stock_news/All_external.csv",
                    help="FNSPID News CSV")
    ap.add_argument("--kaggle-news", default="~/Projects/finBert/kaggle/analyst_ratings_processed.csv",
                    help="Kaggle News CSV (e.g., analyst_ratings_processed.csv or us-equities-news-data CSV)")
    ap.add_argument("--prices", default="~/Projects/finBert/FNSPID/Stock_price/full_history",
                    help="Prices directory (OHLCV CSVs per ticker)")
    ap.add_argument("--out-sentiment", required=False, help="Output path for daily sentiment CSV (used when generating)")
    ap.add_argument("--sentiment-dir", required=False, help="If provided, load precomputed daily sentiment CSV from this path and skip FinBERT.")
    ap.add_argument("--data-source", default="kaggle", choices=["fnspid", "kaggle", "both"],
                    help="Data source: 'fnspid', 'kaggle', or 'both' (merge)")
    ap.add_argument("--ticker", required=True, help="Ticker to train on (single stock) or 'all-tickers' to run all")
    ap.add_argument("--market-tz", default="America/New_York")
    ap.add_argument("--market-close", default="16:00")
    ap.add_argument("--batch-size", type=int, default=8192, help="FinBERT scoring batch size")
    ap.add_argument("--max-length", type=int, default=2048, help="FinBERT tokenizer max_length")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fine-tune", action="store_true", help="Fine-tune FinBERT with NSI labels before scoring")
    ap.add_argument("--use-bodies", action="store_true", help="Include article bodies in sentiment analysis if available")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--load-dir", default=None, help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
    ap.add_argument("--log-batches-debug", action="store_true", help="Log per-batch losses at DEBUG level.")
    ap.add_argument("--use-arima", action="store_true", help="Use hybrid ARIMA-LSTM approach")
    ap.add_argument("--predict-returns", action="store_true", help="Predict returns instead of closing prices")
    ap.add_argument("--dropout-rate", type=float, default=0.2, help="Dropout rate for LSTM layers")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for Adam optimizer")
    ap.add_argument("--x-csv", default=None, help="CSV with X (Twitter) posts, columns: date, text, stock")
    args = ap.parse_args()
    # Expand/normalize user paths
    prices_dir = Path(os.path.expanduser(args.prices))
    fnspid_path = os.path.expanduser(args.fnspid_news)
    kaggle_path = os.path.expanduser(args.kaggle_news)
    load_dir = Path(os.path.expanduser(args.load_dir)).resolve() if args.load_dir else None
    sentiment_path_in = Path(os.path.expanduser(args.sentiment_dir)).resolve() if args.sentiment_dir else None
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    # Output layout
    out_root = _ensure_dir(Path("output") / runtime)
    saved_root = _ensure_dir(out_root / "saved_weights")
    eval_csv_path = out_root / "evaluation.csv"
    # Settings for paper alignment
    s = Settings(
        market_tz=args.market_tz,
        market_close_str=args.market_close,
        batch_size=args.batch_size,
        max_length=args.max_length,
        titles_only=not args.use_bodies,
        dedupe_titles=True,
    )
    start_date = pd.Timestamp(2010, 1, 1, tz='UTC') # Year, Month, Day, UTC timezone
    end_date = pd.Timestamp(2020, 1, 1, tz='UTC') # Align to paper's "up to 2019"
    # Load and filter the news DataFrame (once)
    news_df = load_and_filter_news(fnspid_path, kaggle_path, args.data_source, start_date, end_date, args.use_bodies, args.x_csv)
    logging.info(f"Loaded news dataframe with columns={list(news_df.columns)} shape={news_df.shape}.")
    # Schema
    schema = Schema(
        news_ticker="stock",
        news_time="date",
        news_title="title",
        news_body="body" if "body" in news_df.columns else None,
        price_ticker=None,
        price_date="date",
        price_open="open",
        price_high="high",
        price_low="low",
        price_close="close",
        price_volume="volume",
    )
    # Determine tickers to run
    if args.ticker.lower() == "all-tickers":
        tickers = _list_all_tickers(prices_dir)
        if not tickers:
            raise ValueError(f"No price CSVs found under {prices_dir}")
        logging.info(f"Running all tickers: {len(tickers)} found.")
    else:
        tickers = [args.ticker.upper()]
    # Sentiment daily — either load, or generate once
    if sentiment_path_in and sentiment_path_in.exists():
        sent_daily = pd.read_csv(sentiment_path_in)
        if "trading_date" in sent_daily.columns:
            # make sure trading_date dtype is date (no tz)
            sent_daily["trading_date"] = pd.to_datetime(sent_daily["trading_date"], errors="coerce").dt.date
        logging.info(f"Loaded precomputed sentiment from {sentiment_path_in}")
    else:
        # Build a union trading calendar if running all tickers
        if args.ticker.lower() == "all-tickers":
            # Concatenate just the date column from all price files to form union of trading days
            cal_frames = []
            for t in tickers:
                p = prices_dir / f"{t}.csv"
                if not p.exists():
                    continue
                df_tmp = pd.read_csv(p, usecols=["date"])
                df_tmp["date"] = _ensure_utc(pd.to_datetime(df_tmp["date"], errors="coerce"), s.assume_news_timezone)
                cal_frames.append(df_tmp[["date"]])
            if not cal_frames:
                raise ValueError("No price data to build union trading calendar.")
            prices_calendar_df = pd.concat(cal_frames, ignore_index=True)
        else:
            # Single ticker calendar
            p = prices_dir / f"{tickers[0]}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Price file not found: {p}")
            prices_calendar_df = pd.read_csv(p, parse_dates=["date"], date_format="%Y-%m-%d")
            prices_calendar_df["date"] = _ensure_utc(prices_calendar_df["date"], s.assume_news_timezone)
        # Filter calendar dates to study window
        prices_calendar_df = prices_calendar_df[
            (prices_calendar_df["date"] > start_date) & (prices_calendar_df["date"] < end_date)
        ].dropna(subset=["date"])
        # Where to write generated sentiment
        out_sentiment_csv = args.out_sentiment or str(out_root / "sentiment_daily.csv")
        paths = Paths(str(fnspid_path), str(kaggle_path), str(prices_dir), out_sentiment_csv)
        sent_daily = make_daily_sentiment(news_df, prices_calendar_df, paths, s, schema, args.fine_tune)
    # -------------- Train/Eval per ticker and collect metrics ---------------- #
    eval_rows: List[Dict[str, object]] = []
    for ticker in tickers:
        logging.info("=" * 80)
        logging.info(f"Ticker: {ticker}")
        logging.info("=" * 80)
        price_file = prices_dir / f"{ticker}.csv"
        if not price_file.exists():
            logging.warning(f"Skipping {ticker}: price file not found ({price_file})")
            continue
        prices_df = pd.read_csv(price_file, parse_dates=["date"], date_format="%Y-%m-%d")
        prices_df["date"] = _ensure_utc(prices_df["date"], s.assume_news_timezone)
        prices_df = prices_df[(prices_df["date"] > start_date) & (prices_df["date"] < end_date)].dropna(subset=["date"])
        logging.info(f"Loaded prices for {ticker} with columns={list(prices_df.columns)} shape={prices_df.shape}.")
        # Join
        df_joined = prepare_joined_frame(sent_daily, prices_df, schema, ticker)
        df_supervised, feat_cols = build_supervised_for_ticker(df_joined, ticker=ticker, lookback=args.lookback, predict_returns=args.predict_returns)
        dl_train, dl_val, dl_test, y_scaler, is_hybrid, arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice, original_y_train, original_y_val, original_y_test = make_dataloaders_for_ticker(
            df_supervised, feat_cols, lookback=args.lookback, use_arima=args.use_arima, batch_size=32
        )
        # Prepare saved weights dir for this ticker
        ticker_save_dir = _ensure_dir(saved_root / ticker)
        # Resolve initial load path if provided
        resolved_load_path = None
        if load_dir is not None:
            cand = _find_weight_file_for_ticker(load_dir, ticker)
            if cand:
                resolved_load_path = cand
            else:
                logging.warning(f"--load-dir provided, but no weights found for {ticker} in {load_dir}")
        # Train (now passes y_scaler, returns both scaled and price metrics)
        model, train_hist, val_hist, best_info, price_hist = train_model(
            dl_train, dl_val, input_size=len(feat_cols),
            is_hybrid=is_hybrid,
            arima_pred_train_slice=arima_pred_train_slice,
            arima_pred_val_slice=arima_pred_val_slice,
            original_y_train=original_y_train,
            original_y_val=original_y_val,
            epochs=args.epochs, patience=args.patience, lr=1e-4, weight_decay=args.weight_decay, dropout_rate=args.dropout_rate,
            device=None, # auto-select
            save_dir=ticker_save_dir,
            ticker=ticker,
            load_path=resolved_load_path,
            log_per_batch_debug=args.log_batches_debug,
            y_scaler=y_scaler
        )
        # Evaluate
        metrics = evaluate(model, dl_test, y_scaler, is_hybrid, arima_pred_test_slice, original_y_test)
        logging.info(
            f"[{ticker}] Test (price) MSE={metrics['test_MSE']:.6f} | MAE={metrics['test_MAE']:.6f} | RMSE={metrics['test_RMSE']:.6f} "
            f"(scaled MSE={metrics['test_MSE_scaled']:.6f})"
        )
        if len(train_hist) and len(val_hist):
            logging.info(
                f"[{ticker}] Final (price) Train MSE={price_hist['train_mse'][-1]:.6f} | "
                f"Val MSE={price_hist['val_mse'][-1]:.6f} "
                f"(scaled Train MSE={train_hist[-1]:.6f} | Val MSE={val_hist[-1]:.6f})"
            )
        logging.info(
            f"[{ticker}] Best epoch={best_info['best_epoch']} | "
            f"Best (price) Train MSE={best_info['best_train_mse']:.6f} | "
            f"Best (price) Val MSE={best_info['best_val_mse']:.6f} | Saved at={best_info['best_path']}"
        )
        # Richer training curve CSV
        pd.DataFrame({
            "epoch": np.arange(1, len(train_hist) + 1, dtype=int),
            # scaled losses (training objective)
            "train_mse_scaled": train_hist,
            "val_mse_scaled": val_hist,
            # price-scale metrics (comparable to test)
            "train_mse_price": price_hist["train_mse"],
            "val_mse_price": price_hist["val_mse"],
            "train_mae_price": price_hist["train_mae"],
            "val_mae_price": price_hist["val_mae"],
            "train_rmse_price": price_hist["train_rmse"],
            "val_rmse_price": price_hist["val_rmse"],
        }).to_csv(out_root / f"{ticker}_training_curve.csv", index=False)
        # Evaluation rows (consistent price-scale metrics, plus scaled for reference)
        eval_rows.append({
            "ticker": ticker,
            "best_epoch": best_info["best_epoch"],
            # canonical (price-scale)
            "best_train_mse": best_info["best_train_mse"],
            "best_val_mse": best_info["best_val_mse"],
            "best_train_mae": best_info["best_train_mae"],
            "best_val_mae": best_info["best_val_mae"],
            "best_train_rmse": best_info["best_train_rmse"],
            "best_val_rmse": best_info["best_val_rmse"],
            "saved_weights_path": best_info["best_path"],
            "test_MSE": metrics["test_MSE"],
            "test_MAE": metrics["test_MAE"],
            "test_RMSE": metrics["test_RMSE"],
            # scaled (for reference / debugging)
            "best_train_mse_scaled": best_info["best_train_mse_scaled"],
            "best_val_mse_scaled": best_info["best_val_mse_scaled"],
            "test_MSE_scaled": metrics["test_MSE_scaled"],
            "test_MAE_scaled": metrics["test_MAE_scaled"],
            "test_RMSE_scaled": metrics["test_RMSE_scaled"],
            "epochs_run": best_info["epochs_run"],
            "final_train_mse": best_info["final_train_mse"], # price
            "final_val_mse": best_info["final_val_mse"], # price
            "final_train_mse_scaled": best_info["final_train_mse_scaled"],
            "final_val_mse_scaled": best_info["final_val_mse_scaled"],
        })
        # Optional per-ticker training curve CSV (granular logging artifacts)
        pd.DataFrame({
            "epoch": np.arange(1, len(train_hist) + 1, dtype=int),
            "train_mse": train_hist,
            "val_mse": val_hist,
        }).to_csv(out_root / f"{ticker}_training_curve.csv", index=False)
    # Write evaluation summary
    if eval_rows:
        df_eval = pd.DataFrame(eval_rows)
        df_eval.to_csv(eval_csv_path, index=False)
        logging.info(f"Wrote evaluation summary → {eval_csv_path}")
    else:
        logging.warning("No evaluation rows to write; did all tickers fail to run?")
if __name__ == "__main__":
    main()