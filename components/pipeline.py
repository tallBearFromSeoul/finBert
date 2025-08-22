from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import pandas as pd
import os

from components.sentiment_generator import SentimentGenerator
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from utils.logger import Logger

class Pipeline:
    def __init__(self):
        args, runtime = Pipeline.argparse()
        # Expand/normalize user paths
        prices_dir = Path(os.path.expanduser(args.prices)).resolve()
        fnspid_csv_path = Path(os.path.expanduser(args.fnspid_csv_path)).resolve()
        kaggle_csv_path = Path(os.path.expanduser(args.kaggle_csv_path)).resolve()
        sentiment_csv_path_in = Path(os.path.expanduser(args.sentiment_csv_path_in)).resolve() \
            if args.sentiment_csv_path_in else None
        out_root = Pipeline._ensure_dir(Path("output") / runtime)

        load_dir = Path(os.path.expanduser(args.load_dir)).resolve() if args.load_dir else None
        # Output layout
        saved_root = Pipeline._ensure_dir(out_root / "saved_weights")
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
            start_date, end_date, args.fine_tune)

    @staticmethod
    def _list_all_tickers(prices_dir: Path) -> List[str]:
        return sorted({f.stem.upper() for f in prices_dir.glob("*.csv")})

    @staticmethod
    def _ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

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
            "--fine-tune", action="store_true",
            help="Fine-tune FinBERT with NSI labels before scoring")
        ap.add_argument(
            "--use-bodies", action="store_true",
            help="Include article bodies in sentiment analysis if available")

        ap.add_argument(
            "--load-dir", default=None,
            help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
        ap.add_argument(
            "--log-batches-debug", action="store_true", help="Log per-batch losses at DEBUG level.")
        ap.add_argument(
            "--use-arima", action="store_true", help="Use hybrid ARIMA-LSTM approach")
        ap.add_argument(
            "--predict-returns", action="store_true", help="Predict returns instead of closing prices")
        ap.add_argument(
            "--dropout-rate", type=float, default=0.2, help="Dropout rate for LSTM layers")
        ap.add_argument(
            "--weight-decay", type=float, default=0.01, help="Weight decay for Adam optimizer")
        return ap.parse_args(), runtime

def main():
    Pipeline()

if __name__ == "__main__":
    main()
