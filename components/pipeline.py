from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from yfinance import Ticker
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
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
        if args.seed is not None:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            Logger.info(f"Set deterministic seeds to {args.seed}")
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
            price_close="close",
            price_volume="volume",
        )
        assert isinstance(args.ticker, list), "Sanity check on args.ticker type"
        # Determine tickers to run
        if args.ticker[0].lower() == "all-tickers":
            tickers = Pipeline._list_all_tickers(prices_dir)
            if not tickers:
                raise ValueError(f"No price CSVs found under {prices_dir}")
            Logger.info(f"Running all tickers: {len(tickers)} found.")
        else:
            tickers = args.ticker
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
        df_supervised_list = []
        df_joined_list = []
        ticker_lengths = []
        ticker_joined_lengths = []
        for ticker in tickers_:
            Logger.info(f"Training predictor for ticker: {ticker}")
            price_path = prices_dir_ / f"{ticker}.csv"
            if not price_path.exists():
                Logger.warning(f"Skipping {ticker}: price file not found ({price_path})")
                continue
            ticker_data = Ticker(ticker)
            prices_df = ticker_data.history(start='2005-10-14', end='2023-12-29')
            prices_df.reset_index(inplace=True)
            prices_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                      'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            Logger.info(f"{prices_df.head(5)=} {prices_df.columns=}")
            #prices_df = pd.read_csv(price_path, parse_dates=["date"], date_format="%Y-%m-%d")
            prices_df["date"] = ensure_utc(prices_df["date"])
            prices_df = prices_df[(prices_df["date"] > start_date_) & (prices_df["date"] < end_date_)].dropna(subset=["date"])
            Logger.info(f"Loaded prices for {ticker} with columns={list(prices_df.columns)} shape={prices_df.shape}.")
            # Join
            df_joined = TrainDataPreprocessor.prepare_joined_frame(daily_sentiment_, prices_df, schema_, ticker)
            df_joined_list.append(df_joined)
            ticker_joined_lengths.append(len(df_joined))
            df_supervised, feat_cols = TrainDataPreprocessor.build_supervised_for_ticker(
                df_joined, ticker=ticker, predict_returns=args_.predict_returns
            )
            sentiment_col = 'SentimentScore'
            use_sentiment_in_features = 'finbert' in args_.model.lower()
            df_supervised_list.append(df_supervised)
            ticker_lengths.append(len(df_supervised))
        df_supervised = pd.concat(df_supervised_list, ignore_index=True)
        df_joined = pd.concat(df_joined_list, ignore_index=True)
        sentiment_col = 'SentimentScore'
        use_sentiment_in_features = 'finbert' in args_.model.lower()
        (dl_train, dl_val, dl_test, y_scaler,
         original_y_train, original_y_val, original_y_test,
         dates_train, dates_val, dates_test) = TrainDataPreprocessor.make_dataloaders_for_ticker(
            df_supervised, sentiment_col, use_sentiment_in_features,
            feat_cols, lookback=args_.lookback, batch_size=args_.batch_size,
            scale_method=args_.scale_method, sentiment_threshold=args_.sentiment_threshold
        )
        ticker = "all_tickers"
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
        input_size = len(feat_cols) * args_.lookback + (1 if use_sentiment_in_features else 0)
        model_type = args_.model.split('-')[-1]
        # Train (now passes y_scaler, returns both scaled and price metrics)
        model, train_hist, val_hist, best_info, price_hist = train_model(
            dl_train, dl_val, input_size=input_size,
            original_y_train=original_y_train,
            original_y_val=original_y_val,
            model_type=model_type,
            epochs=args_.epochs, patience=args_.patience,
            lr=args_.lr, weight_decay=args_.weight_decay, dropout_rate=args_.dropout_rate,
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
        dates_full = np.arange(len(dates_train) + len(dates_val) + len(dates_test))
        y_true_full = np.concatenate([y_true_train, y_true_val, y_true_test])
        y_pred_full = np.concatenate([y_pred_train, y_pred_val, y_pred_test])
        train_end_idx = len(y_true_train)
        val_end_idx = train_end_idx + len(y_true_val)
        predictions: Dict[str, Dict[str, np.ndarray]] = {'dates': dates_full,
                                                         'y_true': y_true_full,
                                                         'y_pred': y_pred_full}
        val_start_date = dates_full[train_end_idx] if train_end_idx < len(dates_full) else dates_full[-1]
        test_start_date = dates_full[val_end_idx] if val_end_idx < len(dates_full) else dates_full[-1]
        Pipeline.visualize_full_prediction(
            predictions, target_type, ticker, args_.model, out_root_,
            val_start_date, test_start_date, tickers_
        )
        # Visualize sentiment (full series)
        sentiment_scores = df_joined['SentimentScore'].values
        full_dates = df_joined['trading_date'].values
        Pipeline.visualize_sentiment(sentiment_scores, full_dates, ticker, out_root_, tickers_, ticker_joined_lengths)
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
        with open(out_root_ / f"{ticker}_{args_.model}_training_curve.json", 'w') as f: # CHANGED: Include model in filename
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
                            ticker: str, out_root: Path, tickers: List[str],
                            ticker_lengths: List[int]):
        """
        Visualize daily sentiment scores over time and save as PNG.
        """
        plot_dir = ensure_dir(out_root / "plots")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
        cum_row = 0
        for i, tick in enumerate(tickers):
            length = ticker_lengths[i]
            end_row = cum_row + length
            start_j = cum_row
            end_j = end_row
            if start_j < end_j:
                color = colors[i]
                mask = ~np.isnan(sentiment_scores[start_j:end_j])
                ax.scatter(dates[start_j:end_j][mask], sentiment_scores[start_j:end_j][mask], color=color, label=f'Sentiment {tick}')
            # ax.axvline(dates[end_j - 1] if end_j > 0 else dates[0], color='gray', linestyle='--', label=f'End of {tick}')
            cum_row = end_row
        ax.set_title(f'{ticker} Daily Sentiment')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        plt.gcf().autofmt_xdate()
        plt.savefig(plot_dir / f"{ticker}_sentiment.png")
        plt.close()
        Logger.info(f"Saved sentiment plot for {ticker} to {plot_dir / f'{ticker}_sentiment.png'}")
    @staticmethod
    def visualize_full_prediction(predictions: Dict[str, np.ndarray],
                                  target_type: str, ticker: str, model: str, out_root: Path,
                                  val_start_date, test_start_date, tickers: List[str]):
        """
        Visualize stitched true vs predicted values for train+val+test and save as PNG.
        """
        plot_dir = ensure_dir(out_root / "plots")
        # Main plot with all tickers
        fig, ax = plt.subplots(figsize=(22, 14))
        d = predictions
        ax.scatter(d['dates'], d['y_true'], color='blue', label=f'True', marker='o', s=10, alpha=0.6)
        ax.scatter(d['dates'], d['y_pred'], color='red', linestyle='--', label=f'Predicted', marker='x', s=10, alpha=0.6)
        for date, true, pred in zip(d['dates'], d['y_true'], d['y_pred']):
            ax.vlines(x=date, ymin=min(true, pred), ymax=max(true, pred), color='black', linestyle='-', alpha=0.3)
        ax.axvline(val_start_date, color='green', linestyle='--', label='Start of Validation')
        ax.axvline(test_start_date, color='red', linestyle='--', label='Start of Test')
        ax.set_title(f'{tickers} {model.upper()} Full {target_type.capitalize()} Prediction')
        ax.set_xlabel('Close prices true and predicted')
        ylabel = target_type.capitalize()
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig(plot_dir / f"{ticker}_{model}_full_{target_type}_prediction.png")
        plt.close()
        Logger.info(f"Saved full prediction plot for {tickers} to {plot_dir / f'{ticker}_{model}_full_{target_type}_prediction.png'}")

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
            "--ticker",
            required=True,
            type=str,
            nargs="+",
            help="Ticker(s) to train on (single stock or multiple tickers as a space-separated list, e.g., 'AAPL' or 'AAPL MSFT GOOGL')"
        )
        ap.add_argument(
            "--sentiment-threshold", type=float, default=0.5,
            help="Threshold for absolute sentiment score to consider an article as having sentiment")
        ap.add_argument(
            "--market-tz", default="America/New_York")
        ap.add_argument(
            "--market-close", default="16:30")
        ap.add_argument(
            "--batch-size", type=int, default=1, help="FinBERT scoring batch size")
        ap.add_argument(
            "--max-length", type=int, default=512, help="FinBERT tokenizer max_length")
        ap.add_argument(
            "--lookback", type=int, default=60)
        ap.add_argument(
            "--epochs", type=int, default=1000)
        ap.add_argument(
            "--patience", type=int, default=60)
        ap.add_argument(
            "--use-bodies", action="store_true",
            help="Include article bodies in sentiment analysis if available")
        ap.add_argument(
            "--load-dir", default=None,
            help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
        ap.add_argument(
            "--lr", type=float, default=1e-5, help="Learning rate for Adam optimizer"
        )
        ap.add_argument(
            "--dropout-rate", type=float, default=0.00, help="Dropout rate for LSTM layers")
        ap.add_argument(
            "--weight-decay", type=float, default=0.00, help="Weight decay for Adam optimizer")
        ap.add_argument(
            "--predict-returns", action="store_true", default=False, help="Predict returns instead of closing prices")
        ap.add_argument(
            "--scale-method", default="minmax", choices=["minmax", "standard"],
            help="Use minmax or standard scaling for features")
        ap.add_argument(
            "--model", default="finbert-lstm", choices=["lstm", "rnn", "transformer", "tabmlp", "finbert-lstm", "finbert-rnn", "finbert-transformer", "finbert-tabmlp"],
            help="Model to use: 'lstm', 'rnn', 'transformer', 'tabmlp', or their finbert variants")
        ap.add_argument(
            "--seed", type=int, default=1254, #1254
            help="Preset seed for deterministic training (sets Python, NumPy, and PyTorch seeds)")
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