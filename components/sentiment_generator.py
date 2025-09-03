from __future__ import annotations
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
import polars as pl

from components.article_preprocessor import ArticlePreprocessor
from components.finbert_scorer import FinBertScorer
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.trading_calendar import TradingCalendar
from utils.logger import Logger

class SentimentGenerator:
    def __init__(self, tickers_: List[str], data_paths_: DataPaths, settings_: Settings,
                 schema_: Schema, sentiment_csv_path_in_: Optional[Path], article_df_: pl.LazyFrame,
                 start_date_: datetime, end_date_: datetime,
                 fine_tune_path_: Path, fine_tune_load_path_: Optional[Path]):
        self.tickers: List[str] = tickers_
        self.data_paths: DataPaths = data_paths_
        self.settings: Settings = settings_
        self.schema: Schema = schema_
        self.sentiment_csv_path_in: Optional[Path] = sentiment_csv_path_in_
        self.article_df: pl.LazyFrame = article_df_
        self.start_date: datetime = start_date_
        self.end_date: datetime = end_date_
        self.fine_tune_path: Path = fine_tune_path_
        self.fine_tune_load_path: Optional[Path] = fine_tune_load_path_
        Logger.info(f"Initializing SentimentGenerator with {len(self.tickers)} tickers : running load_or_generate()")
        self._load_or_generate()

    def _load_or_generate(self) -> SentimentGenerator:
        if self.sentiment_csv_path_in:
            self._load()
            return self
        self._generate()
        return self

    def _generate(self) -> None:
        # Build trading calendar (using polars for price data)
        cal_frames = []
        for ticker in self.tickers:
            df_tmp = pl.read_csv(self.data_paths.prices_csv_dir / f"{ticker}.csv", columns=["date", "open", "high", "low", "close", "volume"])
            if df_tmp.is_empty():
                continue
            df_tmp = df_tmp.with_columns(
                pl.lit(ticker).alias("ticker"),
                pl.col("date").str.to_datetime(format="%Y-%m-%d %H:%M:%S %Z", strict=False, time_zone="UTC").alias("date")
            )
            cal_frames.append(df_tmp)
        if not cal_frames:
            raise ValueError("No price data to build union trading calendar.")
        prices_calendar_df = pl.concat(cal_frames, how="vertical_relaxed")

        # Filter calendar dates to study window
        prices_calendar_df = prices_calendar_df.filter(
            (pl.col("date") > self.start_date) & (pl.col("date") < self.end_date)
        )
        self.daily_sentiment = SentimentGenerator._generate_daily_sentiment(
            self.article_df, prices_calendar_df, self.data_paths, self.settings,
            self.schema, self.fine_tune_path, self.fine_tune_load_path
        )

    def _load(self) -> None:
        self.daily_sentiment = pl.read_csv(self.sentiment_csv_path_in)
        if "trading_date" in self.daily_sentiment.schema:
            self.daily_sentiment = self.daily_sentiment.with_columns(
                pl.col("trading_date").str.to_date().alias("trading_date")
            )
        Logger.info(f"Loaded precomputed sentiment from {self.sentiment_csv_path_in}")

    @staticmethod
    def _generate_daily_sentiment(article_df_: pl.LazyFrame, prices_df_: pl.DataFrame,
                                  data_paths_: DataPaths, settings_: Settings, schema_: Schema,
                                  fine_tune_path_: Path, fine_tune_load_path_: Optional[Path]) -> pl.DataFrame:
        calendar = TradingCalendar.build_trading_calendar(prices_df_, schema_)
        Logger.info(f"Built trading calendar with {len(calendar)} days.")
        Logger.info(f"settings: {settings_}\nSchema: {schema_}\nDataPaths: {data_paths_}")
        article_preprocessor = ArticlePreprocessor(settings_, schema_, calendar)
        ticker_col = schema_.article_ticker
        time_col = schema_.article_time
        title_col = schema_.article_title
        Logger.info("Preprocessing articles (lazily with Polars).")

        # Normalize tickers
        article_df_ = article_df_.with_columns(
            article_preprocessor.get_normalize_tickers_expr().alias(ticker_col)
        )
        Logger.info("Normalized tickers")

        # Ensure UTC (fixed logic: check column dtype to decide between replace/convert)
        time_dtype = article_df_.schema[time_col]
        if not isinstance(time_dtype, pl.Datetime):
            raise ValueError(f"Column '{time_col}' must be of type Datetime")
        if time_dtype.time_zone is None:
            # Naive: reinterpret as UTC without adjusting timestamps
            published_utc_expr = pl.col(time_col).dt.replace_time_zone("UTC").alias("published_utc")
        else:
            # Aware: convert to UTC, adjusting timestamps
            published_utc_expr = pl.col(time_col).dt.convert_time_zone("UTC").alias("published_utc")
        article_df_ = article_df_.with_columns(published_utc_expr)
        Logger.info("Converted article times to UTC")

        # Dedupe titles if enabled
        if settings_.dedupe_titles:
            Logger.warning("Performing global deduplication on titles; this may require collecting data.")
            article_df_ = article_df_.unique(subset=[title_col])
            Logger.info("Deduplicated titles")

        # Build text
        article_df_ = article_df_.with_columns(
            article_preprocessor.get_build_text_expr().alias("text")
        )
        Logger.info("Built article texts")

        # Drop NA and filter text length
        article_df_ = article_df_.filter(
            pl.col(ticker_col).is_not_null() & pl.col("published_utc").is_not_null()
        )
        article_df_ = article_df_.filter(pl.col("text").str.len_chars() > 0)

        # Compute trading date lazily (once, for both paths)
        article_df_ = article_df_.with_columns(
            pl.col("published_utc").map_batches(
                lambda x: article_preprocessor.compute_trading_date(x),
                return_dtype=pl.Date
            ).alias("trading_date")
        )
        Logger.info("Computed trading dates for articles.")

        # For fine-tuning (use lazy frame for streaming)
        model = None
        tokenizer = None
        if fine_tune_load_path_ is None:
            Logger.info(f"Fine-tuning FinBERT with NSI labels on full dataset (streaming mode).")
            model, tokenizer = FinBertScorer.fine_tune_finbert(
                article_df_, prices_df_, schema_, settings_, fine_tune_path_, fine_tune_load_path_
            )
            Logger.info("Fine-tuning complete.")
            scorer = FinBertScorer(settings_.batch_size, settings_.max_length, model, tokenizer)
            Logger.info("Scoring articles with fine-tuned FinBERT.")
        else:
            Logger.info(f"Loading fine-tuned FinBERT from {fine_tune_load_path_}.")
            model = AutoModelForSequenceClassification.from_pretrained(fine_tune_load_path_)
            tokenizer = AutoTokenizer.from_pretrained(fine_tune_load_path_)
            scorer = FinBertScorer(settings_.batch_size, settings_.max_length, model, tokenizer)
            Logger.info("Scoring articles with loaded fine-tuned FinBERT.")

        # Score texts (requires collecting in batches due to FinBertScorer)
        def score_texts_in_batches(lazy_df: pl.LazyFrame, batch_size: int = 2**15) -> pl.DataFrame:
            score_dfs = []
            offset = 0
            while True:
                batch_df = lazy_df.slice(offset, batch_size).collect(streaming=True)
                if batch_df.is_empty():
                    break
                # Compute trading_date on the small batch (fast, as it's eager and limited size)
                batch_df = batch_df.with_columns(
                    pl.col("published_utc").map_batches(
                        lambda x: article_preprocessor.compute_trading_date(x),
                        return_dtype=pl.Date
                    ).alias("trading_date")
                )
                texts = batch_df["text"].to_list()
                scores = scorer.score_texts(texts)
                score_df = scores.hstack(batch_df.select([ticker_col, "published_utc", "trading_date"]))
                score_dfs.append(score_df)
                offset += len(batch_df)
            if not score_dfs:
                return pl.DataFrame(schema={
                    "p_pos": pl.Float64,
                    "p_neg": pl.Float64,
                    "p_neu": pl.Float64,
                    "sentiment_score": pl.Float64,
                    "ticker": pl.Utf8,
                    "published_utc": pl.Datetime,
                    "trading_date": pl.Date,
                })
            return pl.concat(score_dfs, how="vertical")

        # Score texts
        article_df_with_scores = score_texts_in_batches(article_df_)
        Logger.info(f"Scored {len(article_df_with_scores)} articles with FinBERT.")

        # Groupby aggregation
        grp = article_df_with_scores.group_by([ticker_col, "trading_date"]).agg(
            N_t=pl.col("sentiment_score").count(),
            SentimentScore=pl.col("sentiment_score").mean(),
            p_pos_mean=pl.col("p_pos").mean(),
            p_neg_mean=pl.col("p_neg").mean(),
            p_neu_mean=pl.col("p_neu").mean(),
        ).rename({ticker_col: "ticker"})

        # Sort and write to CSV
        grp = grp.sort(["ticker", "trading_date"])
        grp.write_csv(data_paths_.out_sentiment_csv_path)
        Logger.info(f"Wrote daily sentiment → {data_paths_.out_sentiment_csv_path}")

        return grp
