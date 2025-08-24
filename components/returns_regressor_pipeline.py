from __future__ import annotations
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import argparse
import json
import lightgbm as lgb
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

from components.sentiment_generator import SentimentGenerator
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.train_data_loader import TrainDataPreprocessor
from components.train import _loss_stats, mse, mae, rmse
from utils.datetime_utils import ensure_utc
from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

class TabDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: Optional[np.ndarray], y: np.ndarray):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long() if X_cat is not None else None
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.X_cat is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_num[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.fc1(x))
        z = self.dropout1(z)
        z = self.relu(self.fc2(z))
        z = self.dropout2(z)
        return self.fc_out(z).squeeze(-1)

class TabMLP(nn.Module):
    def __init__(self, num_features: int, num_tickers: int, embed_dim: int = 32, dropout_rate: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(num_tickers, embed_dim)
        self.fc1 = nn.Linear(num_features + embed_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x_num, x_cat):
        emb = self.embed(x_cat)
        x = torch.cat([x_num, emb], dim=-1)
        z = self.relu(self.fc1(x))
        z = self.dropout1(z)
        z = self.relu(self.fc2(z))
        z = self.dropout2(z)
        return self.fc_out(z).squeeze(-1)



class Pipeline:
    def __init__(self):
        args, runtime = Pipeline.argparse()
        # Expand/normalize user paths
        prices_dir = Path(os.path.expanduser(args.prices_dir)).resolve()
        fnspid_csv_path = Path(os.path.expanduser(args.fnspid_csv_path)).resolve()
        kaggle_csv_path = Path(os.path.expanduser(args.kaggle_csv_path)).resolve()
        sentiment_csv_path_in = Path(os.path.expanduser(args.sentiment_csv_path_in)).resolve() \
            if args.sentiment_csv_path_in else None
        out_root = ensure_dir(Path("output") / runtime)
        out_logs = ensure_dir(out_root / "logs")
        Logger.setup_file(out_logs / "pipeline.log")

        with open(out_root / "run_args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)

        load_dir = Path(os.path.expanduser(args.load_dir)).resolve() if args.load_dir else None
        # Output layout
        saved_root = ensure_dir(out_root / "saved_weights")
        eval_json_path = out_root / "evaluation.json"

        # Settings for paper alignment
        settings = Settings(
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
        article_df = Pipeline.load_and_filter_news(fnspid_csv_path, kaggle_csv_path, args.data_source, start_date, end_date, args.use_bodies)
        Logger.info(f"Loaded news dataframe with columns={list(article_df.columns)} shape={article_df.shape}.")
        # Schema
        schema = Schema(
            article_ticker="stock",
            article_time="date",
            article_title="title",
            article_body="body" if "body" in article_df.columns else None,
            price_date="date",
            price_open="open",
            price_high="high",
            price_low="low",
            price_close="adj close",
            price_volume="volume",
        )
        # Determine tickers to run
        if args.ticker.lower() == "all-tickers":
            tickers = Pipeline._list_all_tickers(prices_dir)
            if not tickers:
                raise ValueError(f"No price CSVs found under {prices_dir}")
            Logger.info(f"Running all tickers: {len(tickers)} found.")
        else:
            tickers = [args.ticker.upper()]

        out_sentiment_csv = out_root / "sentiment_daily.csv"
        data_paths = DataPaths(fnspid_csv_path, kaggle_csv_path, prices_dir, out_sentiment_csv)
        sentiment_generator = SentimentGenerator(
            tickers, data_paths, settings, schema, sentiment_csv_path_in, article_df,
            start_date, end_date, (out_root / args.fine_tune_dir) if args.fine_tune else None)
        self.train(tickers, sentiment_generator.daily_sentiment, prices_dir, schema,
                   start_date, end_date, args, out_root, saved_root, eval_json_path, load_dir)

    # Updated train function
    @staticmethod
    def train(
        tickers_: List[str],
        daily_sentiment_: pd.DataFrame,
        prices_dir_: Path,
        schema_: Schema,
        start_date_: datetime,
        end_date_: datetime,
        args_: argparse.Namespace,
        out_root_: Path,
        saved_root_: Path,
        eval_path_: Path,
        load_dir_: Optional[Path]
    ):
        def prepare_global_data(
            tickers: List[str],
            daily_sentiment: pd.DataFrame,
            prices_dir: Path,
            schema: Schema,
            start_date: datetime,
            end_date: datetime,
            lookback: int
        ) -> pd.DataFrame:
            dfs = []
            for ticker in tickers:
                price_path = prices_dir / f"{ticker}.csv"
                if not price_path.exists():
                    Logger.warning(f"Skipping {ticker}: price file not found ({price_path})")
                    continue
                prices_df = pd.read_csv(price_path, parse_dates=["date"], date_format="%Y-%m-%d")
                prices_df["date"] = ensure_utc(prices_df["date"])
                prices_df = prices_df[
                    (prices_df["date"] > start_date) & (prices_df["date"] < end_date)
                ].dropna(subset=["date"])
                df_joined = TrainDataPreprocessor.prepare_joined_frame(
                    daily_sentiment, prices_df, schema, ticker
                )
                if len(df_joined) < lookback + 2:
                    Logger.warning(f"Skipping {ticker}: insufficient data after join (need >= {lookback + 2} rows)")
                    continue
                # Create lagged features efficiently
                feat_base = ["Open", "High", "Low", "Volume", "Adj Close"]
                lag_dfs = []
                for col in feat_base:
                    # Create a DataFrame with all lags for this column
                    lag_df = pd.DataFrame({
                        f"{col}_lag{lag}": df_joined[col].shift(lag)
                        for lag in range(1, lookback + 1)
                    })
                    lag_dfs.append(lag_df)
                # Concatenate all lagged feature DataFrames at once
                lagged_features = pd.concat(lag_dfs, axis=1)
                # Concatenate the lagged features to the original DataFrame
                df_joined = pd.concat([df_joined, lagged_features], axis=1)
                # Target: NEXT day return
                df_joined["raw_target"] = (
                    df_joined["Adj Close"].shift(-1) - df_joined["Adj Close"]
                ) / df_joined["Adj Close"]
                # Drop rows with NaN (first lookback rows and last row)
                df_joined = df_joined.dropna()
                # Filter to days with actual sentiment (news)
                df_joined = df_joined[df_joined["N_t"] > 0]
                if df_joined.empty:
                    Logger.warning(f"Skipping {ticker}: no days with sentiment after filtering")
                    continue
                df_joined["ticker"] = ticker
                dfs.append(df_joined)
            if not dfs:
                raise ValueError("No data available after processing all tickers")
            big_df = pd.concat(dfs, ignore_index=True)
            big_df = big_df.sort_values("trading_date").reset_index(drop=True)
            Logger.info(f"Prepared global dataframe with shape={big_df.shape}")
            return big_df
        eval_rows: List[Dict[str, object]] = []
        big_df = prepare_global_data(
            tickers_, daily_sentiment_, prices_dir_, schema_, start_date_, end_date_, args_.lookback
        )
        feat_base = ["SentimentScore", "Open", "High", "Low", "Volume", "Adj Close"]
        lag_cols = [f"{col}_lag{lag}" for col in feat_base for lag in range(1, args_.lookback + 1)]
        feat_cols = lag_cols + ["SentimentScore"]  # Include current sentiment, lagged everything including sentiment
        num_feats = len(feat_cols)
        # Temporal split
        n = len(big_df)
        train_all_cut = int(math.floor(n * 0.9))
        train_cut = int(math.floor(train_all_cut * 0.9))  # Val is 10% of train_all
        df_train = big_df.iloc[:train_cut].copy()
        df_val = big_df.iloc[train_cut:train_all_cut].copy()
        df_test = big_df.iloc[train_all_cut:].copy()
        # Scalers for features and target (fit on train only)
        ScalerClass = MinMaxScaler if args_.scale_method == "minmax" else StandardScaler
        x_scaler = ScalerClass()
        x_scaler.fit(df_train[feat_cols])
        df_train.loc[:, feat_cols] = x_scaler.transform(df_train[feat_cols])
        df_val.loc[:, feat_cols] = x_scaler.transform(df_val[feat_cols])
        df_test.loc[:, feat_cols] = x_scaler.transform(df_test[feat_cols])
        y_scaler = ScalerClass()
        y_scaler.fit(df_train[["raw_target"]])
        y_train_s = y_scaler.transform(df_train[["raw_target"]]).ravel()
        y_val_s = y_scaler.transform(df_val[["raw_target"]]).ravel()
        y_test_s = y_scaler.transform(df_test[["raw_target"]]).ravel()
        original_y_train = df_train["raw_target"].values
        original_y_val = df_val["raw_target"].values
        original_y_test = df_test["raw_target"].values
        # Ticker mapping for embedding/categorical
        all_tickers = sorted(big_df["ticker"].unique())
        ticker_to_id = {t: i for i, t in enumerate(all_tickers)}
        num_tickers = len(all_tickers)
        df_train["ticker_id"] = df_train["ticker"].map(ticker_to_id)
        df_val["ticker_id"] = df_val["ticker"].map(ticker_to_id)
        df_test["ticker_id"] = df_test["ticker"].map(ticker_to_id)
        # Model-specific training
        model_name = args_.model
        if model_name == "lgbm":
            lgb_train = lgb.Dataset(
                df_train[feat_cols + ["ticker"]],
                label=y_train_s,
                categorical_feature=["ticker"],
                free_raw_data=False
            )
            lgb_val = lgb.Dataset(
                df_val[feat_cols + ["ticker"]],
                label=y_val_s,
                categorical_feature=["ticker"],
                free_raw_data=False,
                reference=lgb_train
            )
            params = {
                "objective": "regression",
                "metric": "mse",
                "learning_rate": 0.1,
                "verbosity": -1,
            }
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=args_.patience)]
            )
            pred_train_s = model.predict(df_train[feat_cols + ["ticker"]])
            pred_val_s = model.predict(df_val[feat_cols + ["ticker"]])
            pred_test_s = model.predict(df_test[feat_cols + ["ticker"]])
            pred_train = y_scaler.inverse_transform(pred_train_s.reshape(-1, 1)).ravel()
            pred_val = y_scaler.inverse_transform(pred_val_s.reshape(-1, 1)).ravel()
            pred_test = y_scaler.inverse_transform(pred_test_s.reshape(-1, 1)).ravel()
            tr_mse_scaled = mse(y_train_s, pred_train_s)
            val_mse_scaled = mse(y_val_s, pred_val_s)
            test_mse_scaled = mse(y_test_s, pred_test_s)
            tr_mse = mse(original_y_train, pred_train)
            val_mse = mse(original_y_val, pred_val)
            test_mse = mse(original_y_test, pred_test)
            tr_mae = mae(original_y_train, pred_train)
            val_mae = mae(original_y_val, pred_val)
            test_mae = mae(original_y_test, pred_test)
            tr_rmse = rmse(original_y_train, pred_train)
            val_rmse = rmse(original_y_val, pred_val)
            test_rmse = rmse(original_y_test, pred_test)
            save_path = saved_root_ / "lgbm_global.model"
            model.save_model(save_path)
            best_path = str(save_path)
            best_epoch = model.best_iteration
            best_info = {
                "best_epoch": best_epoch,
                "best_val_mse": val_mse,
                "best_train_mse": tr_mse,
                "best_val_mae": val_mae,
                "best_val_rmse": val_rmse,
                "best_train_mae": tr_mae,
                "best_train_rmse": tr_rmse,
                "best_val_mse_scaled": val_mse_scaled,
                "best_train_mse_scaled": tr_mse_scaled,
                "best_path": best_path,
                "epochs_run": best_epoch,
                "final_train_mse": tr_mse,
                "final_val_mse": val_mse,
                "final_train_mse_scaled": tr_mse_scaled,
                "final_val_mse_scaled": val_mse_scaled,
            }
            metrics = {
                "test_MSE": test_mse,
                "test_MAE": test_mae,
                "test_RMSE": test_rmse,
                "test_MSE_scaled": test_mse_scaled,
                "test_MAE_scaled": mae(y_test_s, pred_test_s),
                "test_RMSE_scaled": rmse(y_test_s, pred_test_s),
                "y_true": original_y_test,
                "y_pred": pred_test,
            }
            price_hist = {
                "train_mse": [tr_mse],
                "val_mse": [val_mse],
                "train_mae": [tr_mae],
                "val_mae": [val_mae],
                "train_rmse": [tr_rmse],
                "val_rmse": [val_rmse],
            }
            train_hist_scaled = [tr_mse_scaled]
            val_hist_scaled = [val_mse_scaled]
        else:  # mlp or tabmlp
            is_tab = model_name == "tabmlp"
            cat_col = "ticker_id" if is_tab else None
            ds_train = TabDataset(df_train[feat_cols].values, df_train[cat_col].values if cat_col else None, y_train_s)
            ds_val = TabDataset(df_val[feat_cols].values, df_val[cat_col].values if cat_col else None, y_val_s)
            ds_test = TabDataset(df_test[feat_cols].values, df_test[cat_col].values if cat_col else None, y_test_s)
            dl_train = DataLoader(ds_train, batch_size=args_.batch_size, shuffle=True)
            dl_val = DataLoader(ds_val, batch_size=args_.batch_size, shuffle=False)
            dl_test = DataLoader(ds_test, batch_size=args_.batch_size, shuffle=False)
            input_size = num_feats
            if is_tab:
                model = TabMLP(num_features=input_size, num_tickers=num_tickers, dropout_rate=args_.dropout_rate)
            else:
                model = SimpleMLP(input_size=input_size, dropout_rate=args_.dropout_rate)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            if load_dir_ is not None:
                load_candidates = list(load_dir_.glob("*.pt")) + list(load_dir_.glob("*.pth"))
                if load_candidates:
                    load_path = load_candidates[0]
                    Logger.info(f"Loading initial weights from: {load_path}")
                    state = torch.load(load_path, map_location=device)
                    if isinstance(state, dict) and "state_dict" in state:
                        model.load_state_dict(state["state_dict"])
                    else:
                        model.load_state_dict(state)
                else:
                    Logger.warning(f"No weights found in load_dir: {load_dir_}")
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args_.weight_decay)
            loss_fn = nn.MSELoss()
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
            for ep in range(1, args_.epochs + 1):
                # Train
                model.train()
                tr_losses = []
                tr_pred_s_batches = []
                for batch in dl_train:
                    if is_tab:
                        xb_num, xb_cat, yb = [t.to(device) for t in batch]
                        pred = model(xb_num, xb_cat)
                    else:
                        xb_num, yb = [t.to(device) for t in batch]
                        pred = model(xb_num)
                    loss = loss_fn(pred, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tr_losses.append(float(loss.item()))
                    tr_pred_s_batches.append(pred.detach().cpu().numpy().ravel())
                tr_stats = _loss_stats(tr_losses)
                tr_mse_scaled = tr_stats["mean"]
                if tr_pred_s_batches:
                    y_pred_s = np.concatenate(tr_pred_s_batches)
                    y_pred_price = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                    tr_mse_price = mse(original_y_train, y_pred_price)
                    tr_mae_price = mae(original_y_train, y_pred_price)
                    tr_rmse_price = rmse(original_y_train, y_pred_price)
                else:
                    tr_mse_price = tr_mae_price = tr_rmse_price = float("nan")
                # Val
                model.eval()
                val_losses = []
                val_pred_s_batches = []
                with torch.no_grad():
                    for batch in dl_val:
                        if is_tab:
                            xb_num, xb_cat, yb = [t.to(device) for t in batch]
                            pred = model(xb_num, xb_cat)
                        else:
                            xb_num, yb = [t.to(device) for t in batch]
                            pred = model(xb_num)
                        loss = loss_fn(pred, yb)
                        val_losses.append(float(loss.item()))
                        val_pred_s_batches.append(pred.cpu().numpy().ravel())
                val_stats = _loss_stats(val_losses)
                val_mse_scaled = val_stats["mean"]
                if val_pred_s_batches:
                    y_pred_s = np.concatenate(val_pred_s_batches)
                    y_pred_price = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
                    val_mse_price = mse(original_y_val, y_pred_price)
                    val_mae_price = mae(original_y_val, y_pred_price)
                    val_rmse_price = rmse(original_y_val, y_pred_price)
                else:
                    val_mse_price = val_mae_price = val_rmse_price = float("nan")
                # Histories
                train_hist_scaled.append(tr_mse_scaled)
                val_hist_scaled.append(val_mse_scaled)
                train_hist_price_mse.append(tr_mse_price)
                val_hist_price_mse.append(val_mse_price)
                train_hist_price_mae.append(tr_mae_price)
                val_hist_price_mae.append(val_mae_price)
                train_hist_price_rmse.append(tr_rmse_price)
                val_hist_price_rmse.append(val_rmse_price)
                Logger.info(
                    "Epoch %03d | train (scaled) MSE: mean=%.6f | val (scaled) MSE: mean=%.6f",
                    ep, tr_mse_scaled, val_mse_scaled
                )
                Logger.info(
                    "Epoch %03d | train (price) MSE=%.6f MAE=%.6f RMSE=%.6f | val (price) MSE=%.6f MAE=%.6f RMSE=%.6f",
                    ep, tr_mse_price, tr_mae_price, tr_rmse_price, val_mse_price, val_mae_price, val_rmse_price
                )
                # Early stopping + save best by price-scale val MSE
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
                    prefix = f"{model_name}_global"
                    save_path = saved_root_ / f"{prefix}-best-epoch{ep:03d}.pt"
                    torch.save(best_state, save_path)
                    save_path_for_best = str(save_path.resolve())
                    Logger.info(
                        f"New best (price) val MSE={best_val_price_mse:.6f} at epoch {ep}; saved → {save_path_for_best}"
                    )
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= args_.patience:
                        Logger.info(
                            f"Early stopping at epoch {ep} (best price val MSE={best_val_price_mse:.6f} @ epoch {best_epoch})"
                        )
                        break
            # Load best state
            if best_state is not None:
                model.load_state_dict(best_state)
            # Evaluate on test
            model.eval()
            test_pred_s_batches = []
            with torch.no_grad():
                for batch in dl_test:
                    if is_tab:
                        xb_num, xb_cat, yb = [t.to(device) for t in batch]
                        pred = model(xb_num, xb_cat)
                    else:
                        xb_num, yb = [t.to(device) for t in batch]
                        pred = model(xb_num)
                    test_pred_s_batches.append(pred.cpu().numpy().ravel())
            if test_pred_s_batches:
                pred_test_s = np.concatenate(test_pred_s_batches)
                pred_test = y_scaler.inverse_transform(pred_test_s.reshape(-1, 1)).ravel()
                test_mse = mse(original_y_test, pred_test)
                test_mae = mae(original_y_test, pred_test)
                test_rmse = rmse(original_y_test, pred_test)
                test_mse_scaled = mse(y_test_s, pred_test_s)
                test_mae_scaled = mae(y_test_s, pred_test_s)
                test_rmse_scaled = rmse(y_test_s, pred_test_s)
            else:
                test_mse = test_mae = test_rmse = float("nan")
                test_mse_scaled = test_mae_scaled = test_rmse_scaled = float("nan")
                pred_test = np.array([])
            metrics = {
                "test_MSE": test_mse,
                "test_MAE": test_mae,
                "test_RMSE": test_rmse,
                "test_MSE_scaled": test_mse_scaled,
                "test_MAE_scaled": test_mae_scaled,
                "test_RMSE_scaled": test_rmse_scaled,
                "y_true": original_y_test,
                "y_pred": pred_test,
            }
            best_info = {
                "best_epoch": best_epoch,
                "best_val_mse": best_val_price_mse,
                "best_train_mse": best_train_price_mse,
                "best_val_mae": best_val_price_mae,
                "best_val_rmse": best_val_price_rmse,
                "best_train_mae": best_train_price_mae,
                "best_train_rmse": best_train_price_rmse,
                "best_val_mse_scaled": best_val_scaled_mse,
                "best_train_mse_scaled": best_train_scaled_mse,
                "best_path": save_path_for_best,
                "epochs_run": len(train_hist_scaled),
                "final_train_mse": train_hist_price_mse[-1] if train_hist_price_mse else float("nan"),
                "final_val_mse": val_hist_price_mse[-1] if val_hist_price_mse else float("nan"),
                "final_train_mse_scaled": train_hist_scaled[-1] if train_hist_scaled else float("nan"),
                "final_val_mse_scaled": val_hist_scaled[-1] if val_hist_scaled else float("nan"),
            }
            price_hist = {
                "train_mse": train_hist_price_mse,
                "val_mse": val_hist_price_mse,
                "train_mae": train_hist_price_mae,
                "val_mae": val_hist_price_mae,
                "train_rmse": train_hist_price_rmse,
                "val_rmse": val_hist_price_rmse,
            }

        # Logging
        Logger.info(
            f"Global Test (price) MSE={metrics['test_MSE']:.6f} | MAE={metrics['test_MAE']:.6f} | RMSE={metrics['test_RMSE']:.6f} "
            f"(scaled MSE={metrics['test_MSE_scaled']:.6f})"
        )
        Logger.info(
            f"Global Best epoch={best_info['best_epoch']} | "
            f"Best (price) Train MSE={best_info['best_train_mse']:.6f} | "
            f"Best (price) Val MSE={best_info['best_val_mse']:.6f} | Saved at={best_info['best_path']}"
        )
        # Save training curve if applicable
        if len(train_hist_scaled) > 0:
            curve_data = {
                "epoch": list(range(1, len(train_hist_scaled) + 1)),
                "train_mse_scaled": train_hist_scaled,
                "val_mse_scaled": val_hist_scaled,
                "train_mse_price": price_hist["train_mse"],
                "val_mse_price": price_hist["val_mse"],
                "train_mae_price": price_hist["train_mae"],
                "val_mae_price": price_hist["val_mae"],
                "train_rmse_price": price_hist["train_rmse"],
                "val_rmse_price": price_hist["val_rmse"],
            }
            with open(out_root_ / f"global_{args_.model}_training_curve.json", "w") as f:
                json.dump(curve_data, f, indent=4)
        # Evaluation row for global model
        eval_rows.append({
            "ticker": "global",
            "best_epoch": best_info["best_epoch"],
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
            "best_train_mse_scaled": best_info["best_train_mse_scaled"],
            "best_val_mse_scaled": best_info["best_val_mse_scaled"],
            "test_MSE_scaled": metrics["test_MSE_scaled"],
            "test_MAE_scaled": metrics["test_MAE_scaled"],
            "test_RMSE_scaled": metrics["test_RMSE_scaled"],
            "epochs_run": best_info["epochs_run"],
            "final_train_mse": best_info["final_train_mse"],
            "final_val_mse": best_info["final_val_mse"],
            "final_train_mse_scaled": best_info["final_train_mse_scaled"],
            "final_val_mse_scaled": best_info["final_val_mse_scaled"],
        })
        if eval_rows:
            with open(eval_path_, "w") as f:
                json.dump(eval_rows, f, indent=4)
            Logger.info(f"Wrote evaluation summary → {eval_path_}")
        else:
            Logger.warning("No evaluation rows to write")

    @staticmethod
    def visualize_full_prediction(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray,
                                target_type: str, ticker: str, model: str, out_root: Path,
                                train_end_idx: int, val_end_idx: int):
        """
        Visualize stitched true vs predicted values for train+val+test and save as PNG.
        """
        plot_dir = ensure_dir(out_root / "plots")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, y_true, label='True')
        ax.plot(dates, y_pred, label='Predicted')
        ax.axvline(dates[train_end_idx], color='green', linestyle='--', label='Start of Validation')
        ax.axvline(dates[val_end_idx], color='red', linestyle='--', label='Start of Test')
        ax.set_title(f'{ticker} {model.upper()} Full {target_type.capitalize()} Prediction')
        ax.set_xlabel('Date')
        ylabel = target_type.capitalize()
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.savefig(plot_dir / f"{ticker}_{model}_full_{target_type}_prediction.png")
        plt.close()
        Logger.info(f"Saved full prediction plot for {ticker} to {plot_dir / f'{ticker}_{model}_full_{target_type}_prediction.png'}")

    @staticmethod
    def visualize_sentiment(sentiment_scores: np.ndarray, dates: np.ndarray,
                            ticker: str, out_root: Path):
        """
        Visualize daily sentiment scores over time and save as PNG.
        """
        plot_dir = ensure_dir(out_root / "plots")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, sentiment_scores, label='Sentiment Score')
        ax.set_title(f'{ticker} Daily Sentiment')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        plt.savefig(plot_dir / f"{ticker}_sentiment.png")
        plt.close()
        Logger.info(f"Saved sentiment plot for {ticker} to {plot_dir / f'{ticker}_sentiment.png'}")

    @staticmethod
    def find_weight_file_for_ticker(load_dir: Path, ticker: str) -> Optional[Path]:
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

    @staticmethod
    def _list_all_tickers(prices_dir: Path) -> List[str]:
        return sorted({f.stem.upper() for f in prices_dir.glob("*.csv")})

    @staticmethod
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
        article_df = pd.read_csv(
            fnspid_news,
            usecols=usecols,
            dtype={k: v for k, v in dtypes.items() if v is not None},
            parse_dates=["Date"],
            date_format="%Y-%m-%d"
        )
        return article_df.rename(columns=rename)

    @staticmethod
    def load_kaggle_news(kaggle_news: str, use_bodies: bool) -> pd.DataFrame:
        usecols = ["title", "date", "stock"]
        article_df = pd.read_csv(
            kaggle_news,
            usecols=usecols,
            parse_dates=["date"]
        )
        if use_bodies:
            Logger.warning("No body column found in Kaggle CSV; falling back to titles only.")
        return article_df

    @staticmethod
    def load_and_filter_news(fnspid_news: str, kaggle_news: str, data_source: str,
                            start_date: pd.Timestamp, end_date: pd.Timestamp,
                            use_bodies: bool) -> pd.DataFrame:
        if data_source == "fnspid":
            article_df = Pipeline.load_fnspid_news(fnspid_news, use_bodies)
        elif data_source == "kaggle":
            article_df = Pipeline.load_kaggle_news(kaggle_news, use_bodies)
        else:
            raise ValueError(f"Invalid data_source: {data_source}. Choose 'fnspid' or 'kaggle'.")
        article_df["date"] = pd.to_datetime(article_df["date"], errors="coerce", utc=True)
        article_df = article_df[(article_df["date"] > start_date) & (article_df["date"] < end_date)].dropna(subset=["date"])
        return article_df

    @staticmethod
    def argparse() -> Tuple[argparse.Namespace, str]:
        runtime = datetime.now().strftime("%Y%m%d-%H%M%S")
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--fnspid-csv-path", default="~/Projects/finBert/FNSPID/Stock_news/All_external.csv",
            help="FNSPID News CSV path")
        ap.add_argument(
            "--kaggle-csv-path", default="~/Projects/finBert/kaggle/analyst_ratings_processed.csv",
            help="Kaggle News CSV path")
        ap.add_argument(
            "--prices-dir", default="~/Projects/finBert/FNSPID/Stock_price/full_history",
            help="Prices directory (OHLCV CSVs per ticker)")
        ap.add_argument(
            "--sentiment-csv-path-in", required=False,
            help="If provided, load precomputed daily sentiment CSV from this path and skip FinBERT.")

        ap.add_argument(
            "--fine-tune-dir", default="finetuned_finbert",
            help="Fine-tune FinBERT store path")
        ap.add_argument(
            "--fine-tune", action="store_true",
            help="Fine-tune FinBERT with NSI labels before scoring")

        ap.add_argument(
            "--data-source", default="kaggle", choices=["fnspid", "kaggle"],
            help="Data source: 'fnspid', 'kaggle'")
        ap.add_argument(
            "--ticker", required=True,
            help="Ticker to train on (single stock) or 'all-tickers' to run all")
        ap.add_argument(
            "--market-tz", default="America/New_York")
        ap.add_argument(
            "--market-close", default="16:30")
        ap.add_argument(
            "--batch-size", type=int, default=512, help="FinBERT scoring batch size")
        ap.add_argument(
            "--max-length", type=int, default=512, help="FinBERT tokenizer max_length")
        ap.add_argument(
            "--lookback", type=int, default=60)
        ap.add_argument(
            "--epochs", type=int, default=1000)
        ap.add_argument(
            "--patience", type=int, default=20)
        ap.add_argument(
            "--use-bodies", action="store_true",
            help="Include article bodies in sentiment analysis if available")

        ap.add_argument(
            "--load-dir", default=None,
            help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
        ap.add_argument(
            "--dropout-rate", type=float, default=0.0, help="Dropout rate for LSTM layers")
        ap.add_argument(
            "--weight-decay", type=float, default=0.00, help="Weight decay for Adam optimizer")
        ap.add_argument(
            "--predict-returns", action="store_true", default=False, help="Predict returns instead of closing prices")
        ap.add_argument(
            "--scale-method", default="minmax", choices=["minmax", "standard"],
            help="Use minmax or standard scaling for features")
        ap.add_argument(
            "--model", default="finbert-lstm", choices=["arima", "lstm", "finbert-lstm"],
            help="Model to use: 'arima' (pure ARIMA), 'lstm' (pure LSTM without sentiment), 'finbert_lstm' (LSTM with FinBERT sentiment)")
        return ap.parse_args(), runtime

# standard vs minmax
# model: arima vs lstm vs transformer
# returns vs price
# compare model performance with sentiment vs without sentiment
# plots for data with sentiment vs all data

def main():
    Pipeline()

if __name__ == "__main__":
    main()
