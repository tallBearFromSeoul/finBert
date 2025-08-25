from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf

from components.sentiment_generator import SentimentGenerator
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.train_data_loader import TrainDataPreprocessor
from components.train import evaluate, mse, mae, rmse, train_model
from utils.datetime_utils import ensure_utc
from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

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

    @staticmethod
    def train(
        tickers_: List[str], daily_sentiment_: pd.DataFrame, prices_dir_: Path,
        schema_: Schema, start_date_: datetime, end_date_: datetime, args_: argparse.Namespace,
        out_root_: Path, saved_root_: Path, eval_path_: Path, load_dir_: Optional[Path]):
        eval_rows: List[Dict[str, object]] = []
        for ticker in tickers_:
            Logger.info(f"Training predictor for ticker: {ticker}")
            price_path = prices_dir_ / f"{ticker}.csv"
            if not price_path.exists():
                Logger.warning(f"Skipping {ticker}: price file not found ({price_path})")
                continue
            prices_df = pd.read_csv(price_path, parse_dates=["date"], date_format="%Y-%m-%d")
            prices_df["date"] = ensure_utc(prices_df["date"])
            prices_df = prices_df[(prices_df["date"] > start_date_) & (prices_df["date"] < end_date_)].dropna(subset=["date"])
            Logger.info(f"Loaded prices for {ticker} with columns={list(prices_df.columns)} shape={prices_df.shape}.")
            # Join
            df_joined = TrainDataPreprocessor.prepare_joined_frame(daily_sentiment_, prices_df, schema_, ticker)
            df_supervised, feat_cols = TrainDataPreprocessor.build_supervised_for_ticker(
                df_joined, ticker=ticker, predict_returns=args_.predict_returns
            )
            train_ratio = 0.9
            train_all_len = int(math.floor(len(df_supervised) * train_ratio))
            val_ratio_within_train = 0.1
            train_len = int(math.floor(train_all_len * (1 - val_ratio_within_train)))
            if args_.model == 'arima':
                # Compute splits without lookback/dataloaders
                train_df = df_supervised.iloc[:train_len].copy()
                val_df = df_supervised.iloc[train_len:train_all_len].copy()
                test_df = df_supervised.iloc[train_all_len:].copy()

                use_log = not args_.predict_returns
                if use_log:
                    train_target = np.log(train_df['raw_target'])
                    train_all_target = np.log(df_supervised.iloc[:train_all_len]['raw_target'])
                else:
                    train_target = train_df['raw_target']
                    train_all_target = df_supervised.iloc[:train_all_len]['raw_target']

                # Determine d: differencing order for stationarity
                def get_differencing_order(series, max_d=2):
                    d = 0
                    p_val = adfuller(series)[1]
                    while p_val > 0.05 and d < max_d:
                        series = np.diff(series)
                        p_val = adfuller(series)[1]
                        d += 1
                    return d
                d = get_differencing_order(train_target)
                # Compute ACF and PACF on differenced series
                diff_series = np.diff(train_target, n=d) if d > 0 else train_target
                acf_vals = acf(diff_series, nlags=20, fft=False)
                pacf_vals = pacf(diff_series, nlags=20)
                # Determine q: first lag where ACF cuts off (below confidence interval ~2/sqrt(n))
                conf_int = 2 / np.sqrt(len(diff_series))
                q_candidates = [lag for lag in range(1, 6) if abs(acf_vals[lag]) < conf_int]
                q = q_candidates[0] if q_candidates else 1  # Default to 1 if no clear cutoff
                # Determine p: first lag where PACF cuts off
                p_candidates = [lag for lag in range(1, 6) if abs(pacf_vals[lag]) < conf_int]
                p = p_candidates[0] if p_candidates else 1  # Default to 1 if no clear cutoff
                # Fine-tune with AIC: grid search around candidates
                best_aic = np.inf
                best_order = (p, d, q)
                for pp in range(max(0, p-1), p+2):
                    for qq in range(max(0, q-1), q+2):
                        try:
                            model = ARIMA(train_target, order=(pp, d, qq)).fit()
                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_order = (pp, d, qq)
                        except:
                            continue
                arima_order = best_order
                Logger.info(f"[{ticker}] Selected ARIMA order: {arima_order} based on ACF/PACF and AIC")
                # Fit ARIMA
                arima_order = (1, 1, 2)
                arima_fit = ARIMA(train_target, order=arima_order).fit()
                arima_pred_train = arima_fit.fittedvalues.values
                arima_pred_val = arima_fit.forecast(steps=len(val_df)).values
                arima_fit_full = ARIMA(train_all_target, order=arima_order).fit()
                arima_pred_test = arima_fit_full.forecast(steps=len(test_df)).values
                if use_log:
                    arima_pred_train = np.exp(arima_pred_train)
                    arima_pred_val = np.exp(arima_pred_val)
                    arima_pred_test = np.exp(arima_pred_test)
                else:
                    arima_pred_train = arima_pred_train
                    arima_pred_val = arima_pred_val
                    arima_pred_test = arima_pred_test
                # Compute price-scale metrics (no scaled for ARIMA)
                train_mse = mse(train_df['raw_target'].values, arima_pred_train)
                train_mae = mae(train_df['raw_target'].values, arima_pred_train)
                train_rmse = rmse(train_df['raw_target'].values, arima_pred_train)
                val_mse = mse(val_df['raw_target'].values, arima_pred_val)
                val_mae = mae(val_df['raw_target'].values, arima_pred_val)
                val_rmse = rmse(val_df['raw_target'].values, arima_pred_val)
                test_mse = mse(test_df['raw_target'].values, arima_pred_test)
                test_mae = mae(test_df['raw_target'].values, arima_pred_test)
                test_rmse = rmse(test_df['raw_target'].values, arima_pred_test)
                # Set outputs to match LSTM structure
                model = None
                train_hist = []
                val_hist = []
                best_info = {
                    "best_epoch": 0,
                    "best_train_mse": train_mse,
                    "best_val_mse": val_mse,
                    "best_train_mae": train_mae,
                    "best_val_mae": val_mae,
                    "best_train_rmse": train_rmse,
                    "best_val_rmse": val_rmse,
                    "best_train_mse_scaled": float("nan"),
                    "best_val_mse_scaled": float("nan"),
                    "best_path": "",
                    "epochs_run": 0,
                    "final_train_mse": train_mse,
                    "final_val_mse": val_mse,
                    "final_train_mse_scaled": float("nan"),
                    "final_val_mse_scaled": float("nan"),
                }
                price_hist = {
                    "train_mse": [train_mse],
                    "val_mse": [val_mse],
                    "train_mae": [train_mae],
                    "val_mae": [val_mae],
                    "train_rmse": [train_rmse],
                    "val_rmse": [val_rmse],
                }
                metrics = {
                    "test_MSE": test_mse,
                    "test_MAE": test_mae,
                    "test_RMSE": test_rmse,
                    "test_MSE_scaled": float("nan"),
                    "test_MAE_scaled": float("nan"),
                    "test_RMSE_scaled": float("nan"),
                    "y_true": test_df['raw_target'].values,
                    "y_pred": arima_pred_test,
                }
                lb = 0
                y_true_train = train_df['raw_target'].values
                y_pred_train = arima_pred_train
                y_true_val = val_df['raw_target'].values
                y_pred_val = arima_pred_val
                y_true_test = test_df['raw_target'].values
                y_pred_test = arima_pred_test
                dates_train = train_df['trading_date'].values
                dates_val = val_df['trading_date'].values
                dates_test = test_df['trading_date'].values
                # No training curve for ARIMA
                # Test dates without lookback
                test_start_idx = train_all_len
                test_dates = df_supervised['trading_date'].iloc[test_start_idx:].values
                # Log
                Logger.info(
                    f"[{ticker}] Train (price) MSE={train_mse:.6f} | MAE={train_mae:.6f} | RMSE={train_rmse:.6f}"
                )
                Logger.info(
                    f"[{ticker}] Val (price) MSE={val_mse:.6f} | MAE={val_mae:.6f} | RMSE={val_rmse:.6f}"
                )
                Logger.info(
                    f"[{ticker}] Test (price) MSE={test_mse:.6f} | MAE={test_mae:.6f} | RMSE={test_rmse:.6f}"
                )
            else:
                # LSTM variants
                sentiment_col: Optional[str] = 'SentimentScore' if args_.model == 'finbert-lstm' else None
                (dl_train, dl_val, dl_test, y_scaler,
                 original_y_train, original_y_val, original_y_test,
                 dates_train, dates_val, dates_test) = TrainDataPreprocessor.make_dataloaders_for_ticker(
                    df_supervised, sentiment_col,
                    feat_cols, lookback=args_.lookback, batch_size=args_.batch_size,
                    scale_method=args_.scale_method
                )

                # Prepare saved weights dir for this ticker
                ticker_save_dir = ensure_dir(saved_root_ / ticker)
                # Resolve initial load path if provided
                resolved_load_path = None
                if load_dir_ is not None:
                    cand = Pipeline.find_weight_file_for_ticker(load_dir_, ticker)
                    if cand:
                        resolved_load_path = cand
                        Logger.info(f"--load-dir provided, found weights {ticker} {cand} in {load_dir_}")
                    else:
                        Logger.warning(f"--load-dir provided, but no weights found for {ticker} in {load_dir_}")
                input_size = len(feat_cols) * args_.lookback + (1 if sentiment_col else 0)
                # Train (now passes y_scaler, returns both scaled and price metrics)
                model, train_hist, val_hist, best_info, price_hist = train_model(
                    dl_train, dl_val, input_size=input_size,
                    original_y_train=original_y_train,
                    original_y_val=original_y_val,
                    epochs=args_.epochs, patience=args_.patience,
                    lookback=args_.lookback,
                    lr=1e-4, weight_decay=args_.weight_decay, dropout_rate=args_.dropout_rate,
                    save_dir=ticker_save_dir,
                    ticker=ticker,
                    load_path=resolved_load_path,
                    y_scaler=y_scaler
                )
                # Evaluate
                metrics_train = evaluate(model, dl_train, y_scaler, original_y_train)
                metrics_val = evaluate(model, dl_val, y_scaler, original_y_val)
                metrics_test = evaluate(model, dl_test, y_scaler, original_y_test)
                metrics = metrics_test
                lb = args_.lookback
                y_true_train = metrics_train['y_true']
                y_pred_train = metrics_train['y_pred']
                y_true_val = metrics_val['y_true']
                y_pred_val = metrics_val['y_pred']
                y_true_test = metrics_test['y_true']
                y_pred_test = metrics_test['y_pred']

            # Determine target type for prediction plot
            if args_.predict_returns:
                target_type = 'returns'
            else:
                target_type = 'price'
            # Visualize predictions
            dates_full = np.concatenate([dates_train, dates_val, dates_test])
            y_true_full = np.concatenate([y_true_train, y_true_val, y_true_test])
            y_pred_full = np.concatenate([y_pred_train, y_pred_val, y_pred_test])
            Pipeline.visualize_full_prediction(
                y_true_full, y_pred_full, dates_full, target_type, ticker, args_.model, out_root_,
                len(y_true_train), len(y_true_train) + len(y_true_val)
            )
            # Visualize sentiment (full series)
            sentiment_scores = df_joined['SentimentScore'].values
            full_dates = df_joined['trading_date'].values
            Pipeline.visualize_sentiment(sentiment_scores, full_dates, ticker, out_root_)
            Logger.info(
                f"[{ticker}] Test (price) MSE={metrics['test_MSE']:.6f} | MAE={metrics['test_MAE']:.6f} | RMSE={metrics['test_RMSE']:.6f} "
                f"(scaled MSE={metrics['test_MSE_scaled']:.6f})"
            )
            if len(train_hist) and len(val_hist):
                Logger.info(
                    f"[{ticker}] Final (price) Train MSE={price_hist['train_mse'][-1]:.6f} | "
                    f"Val MSE={price_hist['val_mse'][-1]:.6f} "
                    f"(scaled Train MSE={train_hist[-1]:.6f} | Val MSE={val_hist[-1]:.6f})"
                )
            Logger.info(
                f"[{ticker}] Best epoch={best_info['best_epoch']} | "
                f"Best (price) Train MSE={best_info['best_train_mse']:.6f} | "
                f"Best (price) Val MSE={best_info['best_val_mse']:.6f} | Saved at={best_info['best_path']}"
            )
            if args_.model != 'arima':
                curve_data = {
                    "epoch": np.arange(1, len(train_hist) + 1, dtype=int).tolist(),
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
                }
                with open(out_root_ / f"{ticker}_{args_.model}_training_curve.json", 'w') as f:  # CHANGED: Include model in filename
                    json.dump(curve_data, f, indent=4)
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
        if eval_rows:
            with open(eval_path_, 'w') as f:
                json.dump(eval_rows, f, indent=4)
            Logger.info(f"Wrote evaluation summary → {eval_path_}")
        else:
            Logger.warning("No evaluation rows to write; did all tickers_ fail to run?")

    @staticmethod
    def visualize_sentiment(sentiment_scores: np.ndarray, dates: np.ndarray,
                            ticker: str, out_root: Path):
        """
        Visualize daily sentiment scores over time and save as PNG.
        """
        mask = ~np.isnan(sentiment_scores)
        sentiment_scores = sentiment_scores[mask]
        dates = dates[mask]
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
            "--batch-size", type=int, default=8, help="FinBERT scoring batch size")
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
