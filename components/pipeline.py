from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
import pandas as pd
import os

from components.sentiment_generator import SentimentGenerator
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.train_data_loader import TrainDataLoader, TrainDataPreprocessor
from components.train import evaluate, train_model
from utils.datetime_utils import ensure_utc
from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

class Pipeline:
    def __init__(self):
        args, runtime = Pipeline.argparse()
        # Expand/normalize user paths
        prices_dir = Path(os.path.expanduser(args.prices)).resolve()
        fnspid_csv_path = Path(os.path.expanduser(args.fnspid_csv_path)).resolve()
        kaggle_csv_path = Path(os.path.expanduser(args.kaggle_csv_path)).resolve()
        sentiment_csv_path_in = Path(os.path.expanduser(args.sentiment_csv_path_in)).resolve() \
            if args.sentiment_csv_path_in else None
        out_root = ensure_dir(Path("output") / runtime)

        load_dir = Path(os.path.expanduser(args.load_dir)).resolve() if args.load_dir else None
        # Output layout
        saved_root = ensure_dir(out_root / "saved_weights")
        eval_csv_path = out_root / "evaluation.csv"

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
            price_close="close",
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
                   start_date, end_date, args, out_root, saved_root, eval_csv_path, load_dir)

    @staticmethod
    def train(
        tickers_: List[str], daily_sentiment_: pd.DataFrame, prices_dir_: Path,
        schema_: Schema, start_date_: datetime, end_date_: datetime, args_: argparse.Namespace,
        out_root_: Path, saved_root_: Path, eval_csv_path_: Path, load_dir_: Optional[Path]):
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
                df_joined, ticker=ticker, lookback=args_.lookback, predict_returns=args_.predict_returns)
            (dl_train, dl_val, dl_test, y_scaler, is_hybrid,
             arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice,
             original_y_train, original_y_val, original_y_test) = TrainDataPreprocessor.make_dataloaders_for_ticker(
                df_supervised, feat_cols, lookback=args_.lookback, use_arima=args_.use_arima, batch_size=args_.batch_size)

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
            # Train (now passes y_scaler, returns both scaled and price metrics)
            model, train_hist, val_hist, best_info, price_hist = train_model(
                dl_train, dl_val, input_size=len(feat_cols),
                is_hybrid=is_hybrid,
                arima_pred_train_slice=arima_pred_train_slice,
                arima_pred_val_slice=arima_pred_val_slice,
                original_y_train=original_y_train,
                original_y_val=original_y_val,
                epochs=args_.epochs, patience=args_.patience, lr=1e-4, weight_decay=args_.weight_decay, dropout_rate=args_.dropout_rate,
                device=None, # auto-select
                save_dir=ticker_save_dir,
                ticker=ticker,
                load_path=resolved_load_path,
                y_scaler=y_scaler
            )
            # Evaluate
            metrics = evaluate(model, dl_test, y_scaler, is_hybrid, arima_pred_test_slice, original_y_test)
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
            }).to_csv(out_root_ / f"{ticker}_training_curve.csv", index=False)
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
            # Optional per-ticker training curve CSV (granular Logger artifacts)
            pd.DataFrame({
                "epoch": np.arange(1, len(train_hist) + 1, dtype=int),
                "train_mse": train_hist,
                "val_mse": val_hist,
            }).to_csv(out_root_ / f"{ticker}_training_curve.csv", index=False)
        # Write evaluation summary
        if eval_rows:
            df_eval = pd.DataFrame(eval_rows)
            df_eval.to_csv(eval_csv_path_, index=False)
            Logger.info(f"Wrote evaluation summary → {eval_csv_path_}")
        else:
            Logger.warning("No evaluation rows to write; did all tickers_ fail to run?")

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
            "--prices", default="~/Projects/finBert/FNSPID/Stock_price/full_history",
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
            "--market-close", default="16:00")
        ap.add_argument(
            "--batch-size", type=int, default=8192, help="FinBERT scoring batch size")
        ap.add_argument(
            "--max-length", type=int, default=512, help="FinBERT tokenizer max_length")
        ap.add_argument(
            "--lookback", type=int, default=60)
        ap.add_argument(
            "--epochs", type=int, default=100)
        ap.add_argument(
            "--patience", type=int, default=20)
        ap.add_argument(
            "--use-bodies", action="store_true",
            help="Include article bodies in sentiment analysis if available")

        ap.add_argument(
            "--load-dir", default=None,
            help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
        ap.add_argument(
            "--use-arima", action="store_true", default=True, help="Use hybrid ARIMA-LSTM approach")
        ap.add_argument(
            "--predict-returns", action="store_true", default=True, help="Predict returns instead of closing prices")
        ap.add_argument(
            "--dropout-rate", type=float, default=0.2, help="Dropout rate for LSTM layers")
        ap.add_argument(
            "--weight-decay", type=float, default=0.01, help="Weight decay for Adam optimizer")
        return ap.parse_args(), runtime

def main():
    Pipeline()

if __name__ == "__main__":
    main()
