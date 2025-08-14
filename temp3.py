from __future__ import annotations
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
# ----------------------------- Config & Utilities ----------------------------- #
@dataclass(frozen=True)
class Paths:
    news_csv: str
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
@dataclass(frozen=True)
class Settings:
    market_tz: str = "America/New_York"
    market_close_str: str = "16:00"
    batch_size: int = 1024
    max_length: int = 1024 # tokenizer truncation
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
    def __init__(self, batch_size: int = 1024, max_length: int = 1024):
        model_id = "ProsusAI/finbert"
        device = 0 if torch.cuda.is_available() else -1
        self.batch_size = batch_size
        self.pipe = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_id),
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            device=device,
            return_all_scores=True,
            truncation=True
        )
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            self.pipe.model = torch.compile(self.pipe.model)  # Compile the model
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
                with autocast():  # Enable mixed precision
                    outputs = self.pipe.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for prob in probs:
                    p_pos, p_neg, p_neu = prob
                    results.append((p_pos, p_neg, p_neu, p_pos - p_neg))

        return pd.DataFrame(results, columns=["p_pos", "p_neg", "p_neu", "sentiment_score"])
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
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.fc1 = nn.Linear(100, 25)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(25, 1)
    def forward(self, x):
        # x: (B, T, F)
        out1, _ = self.lstm1(x) # (B, T, 100)
        out2, _ = self.lstm2(out1) # (B, T, 100)
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
def build_supervised_for_ticker(df_all: pd.DataFrame, ticker: str, lookback: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble features (no scaling here) and keep raw Close for later target scaling.
    """
    df = df_all[df_all["ticker"] == ticker].copy().sort_values("trading_date").reset_index(drop=True)
    feat_cols = ["SentimentScore"]
    for c in ["Open", "High", "Low", "Volume", "Close"]:
        if c in df.columns:
            feat_cols.append(c)
    df["Close_raw"] = df["Close"].astype(float)
    return df, feat_cols
def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(math.floor(n * train_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
def split_train_val(train_df: pd.DataFrame, val_ratio_within_train: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    cut = int(math.floor(n * (1 - val_ratio_within_train)))
    return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()
def make_dataloaders_for_ticker(df: pd.DataFrame, feat_cols: List[str], lookback: int, batch_size: int = 1):
    """
    Chronological split; fit scalers on TRAIN only; transform all splits; build PyTorch loaders.
    """
    # Splits
    train_all, test = temporal_split(df, train_ratio=0.8)
    train, val = split_train_val(train_all, val_ratio_within_train=0.1)
    # Fit X scaler on TRAIN only
    x_scaler = MinMaxScaler()
    x_scaler.fit(train[feat_cols].astype(float))
    # Transform X across splits
    for part in [train, val, test]:
        part.loc[:, feat_cols] = x_scaler.transform(part[feat_cols].astype(float))
    # Fit y scaler on TRAIN only
    y_scaler = MinMaxScaler()
    y_scaler.fit(train[["Close_raw"]].astype(float))
    # Add scaled y
    for part in [train, val, test]:
        part["y"] = y_scaler.transform(part[["Close_raw"]].astype(float)).astype(float)
    # Datasets
    ds_train = SequenceDataset(train, feat_cols, "y", lookback)
    ds_val = SequenceDataset(val, feat_cols, "y", lookback)
    ds_test = SequenceDataset(test, feat_cols, "y", lookback)
    # Loaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    return dl_train, dl_val, dl_test, y_scaler
# ----------------------------- End-to-end ----------------------------------- #
def make_daily_sentiment(news_df: pd.DataFrame, prices_df: pd.DataFrame,
                         paths: Paths, s: Settings, schema: Schema) -> pd.DataFrame:
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
    scorer = FinBertScorer(batch_size=s.batch_size, max_length=s.max_length)
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
    agg = grp.sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(paths.out_sentiment_csv) or ".", exist_ok=True)
    agg.to_csv(paths.out_sentiment_csv, index=False)
    logging.info(f"Wrote daily sentiment → {paths.out_sentiment_csv}")
    return agg
def train_model(dl_train, dl_val, input_size: int, epochs: int = 100, patience: int = 20, lr: float = 1e-3,
                device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(input_size=input_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, float("inf")
    bad_epochs = 0
    train_hist, val_hist = [], []
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tr_losses = []
        for xb, yb in dl_train:
            xb = xb.to(device) # (B, T, F)
            yb = yb.to(device) # (B,)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_mse = float(np.mean(tr_losses)) if tr_losses else float("nan")
        # ---- Val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())
        val_mse = float(np.mean(val_losses)) if val_losses else float("nan")
        train_hist.append(tr_mse); val_hist.append(val_mse)
        logging.info(f"Epoch {ep:03d} | train MSE={tr_mse:.6f} | val MSE={val_mse:.6f}")
        # Early stopping
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(f"Early stopping at epoch {ep} (best val MSE={best_val:.6f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_hist, val_hist
def evaluate(model: nn.Module, dl_test: DataLoader, y_scaler: MinMaxScaler, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds_scaled, trues_scaled = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().ravel()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy().ravel())
    y_pred_s = np.concatenate(preds_scaled)
    y_true_s = np.concatenate(trues_scaled)
    # Inverse-transform to price scale
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_true_s.reshape(-1, 1)).ravel()
    return {
        "test_MSE": mse(y_true, y_pred),
        "test_MAE": mae(y_true, y_pred),
        "test_RMSE": rmse(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }
# ----------------------------------- Main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="FNSPID/Stock_news/All_external.csv",
                    help="News CSV")
    ap.add_argument("--prices", default="FNSPID/Stock_price/full_history",
                    help="Prices directory (OHLCV CSVs per ticker)")
    ap.add_argument("--out-sentiment", required=True, help="Output path for daily sentiment CSV")
    ap.add_argument("--ticker", required=True, help="Ticker to train on (single stock, as in paper)")
    ap.add_argument("--market-tz", default="America/New_York")
    ap.add_argument("--market-close", default="16:00")
    ap.add_argument("--batch-size", type=int, default=32, help="FinBERT scoring batch size")
    ap.add_argument("--max-length", type=int, default=512, help="FinBERT tokenizer max_length")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    # Settings for paper alignment
    s = Settings(
        market_tz=args.market_tz,
        market_close_str=args.market_close,
        batch_size=args.batch_size,
        max_length=args.max_length,
        titles_only=True,
        dedupe_titles=True,
    )
    dtypes = {
        "Article_title": str,
        "Stock_symbol": str
    }
    start_date = pd.Timestamp(2015, 1, 1, tz='UTC')  # Year, Month, Day, UTC timezone
    end_date = pd.Timestamp(2020, 12, 31, tz='UTC')  # Year, Month, Day, UTC timezone
    # Load and filter the DataFrame
    news_df = pd.read_csv(
        args.news,
        usecols=["Date", "Article_title", "Stock_symbol"],
        dtype=dtypes,
        parse_dates=["Date"],
        date_format="%Y-%m-%d"
    )
    # Ensure Date column is datetime
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
    news_df = news_df[(news_df["Date"] > start_date) & (news_df["Date"] < end_date)].dropna(subset=["Date"])
    logging.info(f"loaded news dataframe with {news_df.columns} and {news_df.shape}.")
    # Load and filter the prices DataFrame
    prices_df = pd.read_csv(
        f"{args.prices}/{args.ticker}.csv",
        parse_dates=["date"],
        date_format="%Y-%m-%d"
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce")
    prices_df["date"] = _ensure_utc(prices_df["date"], s.assume_news_timezone)  # Convert to UTC
    prices_df = prices_df[(prices_df["date"] > start_date) & (prices_df["date"] < end_date)].dropna(subset=["date"])
    logging.info(f"loaded prices dataframe with {prices_df.columns} and {prices_df.shape}.")
    schema = Schema(
        news_ticker="Stock_symbol",
        news_time="Date",
        news_title="Article_title",
        news_body=None,
        price_ticker=None,
        price_date="date",
        price_open="open",
        price_high="high",
        price_low="low",
        price_close="close",
        price_volume="volume",
    )
    # Step 1: sentiment per (ticker, day)
    paths = Paths(args.news, args.prices, args.out_sentiment)
    sent_daily = make_daily_sentiment(news_df, prices_df, paths, s, schema)
    # Step 2: join with prices, build features for the requested ticker
    df_joined = prepare_joined_frame(sent_daily, prices_df, schema, args.ticker.upper())
    df_supervised, feat_cols = build_supervised_for_ticker(
        df_joined, ticker=args.ticker.upper(), lookback=args.lookback)
    # Step 3: dataloaders (strict chronological 80/20 split; val is last 10% of train)
    dl_train, dl_val, dl_test, y_scaler = make_dataloaders_for_ticker(
        df_supervised, feat_cols, lookback=args.lookback, batch_size=1
    )
    # Step 4: train
    model, train_hist, val_hist = train_model(
        dl_train, dl_val, input_size=len(feat_cols),
        epochs=args.epochs, patience=args.patience, lr=1e-3
    )
    # Step 5: evaluate with MSE/MAE/RMSE on test (inverse-transformed)
    metrics = evaluate(model, dl_test, y_scaler)
    logging.info(
        f"Test MSE={metrics['test_MSE']:.6f} | MAE={metrics['test_MAE']:.6f} | RMSE={metrics['test_RMSE']:.6f}")
    if len(train_hist) and len(val_hist):
        logging.info(f"Final Train MSE={train_hist[-1]:.6f} | Final Val MSE={val_hist[-1]:.6f}")

if __name__ == "__main__":
    main()