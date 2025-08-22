from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

from components.article_preprocessor import ArticlePreprocessor
from components.finbert_scorer import FinBertScorer
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.trading_calendar import TradingCalendar
from utils.datetime_utils import ensure_utc
from utils.logger import Logger

class SentimentGenerator:
    def __init__(self, tickers_: List[str], data_paths_: DataPaths, settings_: Settings,
                 schema_: Schema, sentiment_csv_path_in_: Optional[Path], article_df_: pd.DataFrame,
                 start_date_: datetime, end_date_: datetime, fine_tune_: bool):
        self.tickers: List[str] = tickers_
        self.data_paths: DataPaths = data_paths_
        self.settings: Settings = settings_
        self.schema: Schema = schema_
        self.sentiment_csv_path_in: Optional[Path] = sentiment_csv_path_in_
        self.article_df: pd.DataFrame = article_df_
        self.start_date: datetime = start_date_
        self.end_date: datetime = end_date_
        self.fine_tune: bool = fine_tune_
        self._load_or_generate()

    def _load_or_generate(self) -> SentimentGenerator:
        # Sentiment daily — either load, or generate once
        if self.sentiment_csv_path_in:
            self._load()
            return self
        self._generate()
        return self

    def _generate(self) -> None:
        # Build a union trading calendar
        # Concatenate just the date column from all price files to form union of trading days
        cal_frames = []
        for ticker in self.tickers:
            df_tmp = pd.read_csv(self.data_paths.prices_csv_dir / f"{ticker}.csv", usecols=["date"])
            df_tmp["date"] = ensure_utc(pd.to_datetime(df_tmp["date"], errors="coerce"))
            cal_frames.append(df_tmp[["date"]])
        if not cal_frames:
            raise ValueError("No price data to build union trading calendar.")
        prices_calendar_df = pd.concat(cal_frames, ignore_index=True)

        # Filter calendar dates to study window
        prices_calendar_df = prices_calendar_df[
            (prices_calendar_df["date"] > self.start_date)
            & (prices_calendar_df["date"] < self.end_date)].dropna(subset=["date"])
        # Where to write generated sentiment
        self.daily_sentiment = SentimentGenerator._generate_daily_sentiment(self.article_df,
                                                                            prices_calendar_df,
                                                                            self.data_paths,
                                                                            self.settings,
                                                                            self.schema,
                                                                            self.fine_tune)

    def _load(self) -> None:
        self.daily_sentiment = pd.read_csv(self.sentiment_csv_path_in)
        if "trading_date" in self.daily_sentiment.columns:
            # make sure trading_date dtype is date (no tz)
            self.daily_sentiment["trading_date"] = \
                pd.to_datetime(self.daily_sentiment["trading_date"], errors="coerce").dt.date
        Logger.info(f"Loaded precomputed sentiment from {self.sentiment_csv_path_in}")

    @staticmethod
    def _generate_daily_sentiment(article_df_: pd.DataFrame, prices_df_: pd.DataFrame,
                                  data_paths_: DataPaths, settings_: Settings, schema_: Schema,
                                  fine_tune_: bool) -> pd.DataFrame:
        calendar = TradingCalendar.build_trading_calendar(prices_df_, schema_)
        article_preprocessor = ArticlePreprocessor(settings_, schema_, calendar)
        ticker_col = schema_.article_ticker
        time_col = schema_.article_time
        title_col = schema_.article_title

        article_df_[ticker_col] = article_preprocessor.normalize_tickers(article_df_)
        article_df_["published_utc"] = ensure_utc(article_df_[time_col])
        if settings_.dedupe_titles:
            article_df_ = article_df_.drop_duplicates(subset=[title_col]).reset_index(drop=True)
        article_df_["text"] = article_preprocessor.build_text(article_df_)
        article_df_ = article_df_.dropna(subset=[ticker_col, "published_utc"]).reset_index(drop=True)
        article_df_ = article_df_[article_df_["text"].str.len() > 0].reset_index(drop=True)
        model = None
        if fine_tune_:
            model, tokenizer = FinBertScorer.fine_tune_finbert(article_df_, prices_df_, schema_,
                                                               settings_, article_preprocessor)
            scorer = FinBertScorer(settings_.batch_size, settings_.max_length,
                                   model=model, tokenizer=tokenizer)
        else:
            scorer = FinBertScorer(settings_.batch_size, settings_.max_length)
        score_df = scorer.score_texts(article_df_["text"].tolist())
        assert len(score_df) == len(article_df_), "Score length mismatch."
        article_df_ = pd.concat([article_df_, score_df], axis=1)
        article_df_["trading_date"] = article_preprocessor.compute_trading_date(
            article_df_["published_utc"])
        grp = article_df_.groupby([ticker_col, "trading_date"], as_index=False).agg(
            N_t=("sentiment_score", "size"),
            SentimentScore=("sentiment_score", "mean"),
            p_pos_mean=("p_pos", "mean"),
            p_neg_mean=("p_neg", "mean"),
            p_neu_mean=("p_neu", "mean"),
        )
        grp = grp.rename(columns={ticker_col: "ticker"})
        grp["trading_date"] = pd.to_datetime(grp["trading_date"]).dt.date
        grp.sort_values(["ticker", "trading_date"]).to_csv(data_paths_.out_sentiment_csv_path, index=False)
        Logger.info(f"Wrote daily sentiment → {data_paths_.out_sentiment_csv_path}")
        return grp

