from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
import os

from components.article_preprocessor import ArticlePreprocessor
from components.finbert_scorer import FinBertScorer
from components.types import Paths, Schema, Settings, TradingCalendar
from utils.datetime import ensure_utc
from utils.logger import Logger

class SentimentGenerator:
    def __init__(self, tickers_: List[str], paths_: Paths, sentiment_path_in_: Optional[Path]):
        self.tickers: List[str] = tickers_
        self.paths: Paths = paths_
        self.sentiment_path_in: Optional[Path] = sentiment_path_in_

    def load_or_generate(self, start_date_: datetime, end_date_: datetime) -> SentimentGenerator:
        # Sentiment daily — either load, or generate once
        if self.sentiment_path_in and self.sentiment_path_in.exists():
            self.load(self.sentiment_path_in)
            return self
        self.generate(start_date_, end_date_)
        return self

    def generate(self, article_df_: pd.DataFrame, settings_: Settings, schema_: Schema,
                 start_date_: datetime, end_date_: datetime, fine_tune_: bool) -> None:
        # Build a union trading calendar
        # Concatenate just the date column from all price files to form union of trading days
        cal_frames = []
        for t in self.tickers:
            p = self.prices_path_in / f"{t}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Price file not found: {p}")
            df_tmp = pd.read_csv(p, usecols=["date"])
            df_tmp["date"] = ensure_utc(pd.to_datetime(df_tmp["date"], errors="coerce"))
            cal_frames.append(df_tmp[["date"]])
        if not cal_frames:
            raise ValueError("No price data to build union trading calendar.")
        prices_calendar_df = pd.concat(cal_frames, ignore_index=True)

        # Filter calendar dates to study window
        prices_calendar_df = prices_calendar_df[
            (prices_calendar_df["date"] > start_date_) & (prices_calendar_df["date"] < end_date_)
        ].dropna(subset=["date"])
        # Where to write generated sentiment
        self.daily_sentiment = SentimentGenerator.generate_daily_sentiment(article_df_,
                                                                           prices_calendar_df,
                                                                           self.paths,
                                                                           settings_,
                                                                           schema_,
                                                                           fine_tune_)

    def load(self) -> None:
        self.daily_sentiment = pd.read_csv(self.sentiment_path_in)
        if "trading_date" in self.daily_sentiment.columns:
            # make sure trading_date dtype is date (no tz)
            self.daily_sentiment["trading_date"] = \
                pd.to_datetime(self.daily_sentiment["trading_date"], errors="coerce").dt.date
        Logger.info(f"Loaded precomputed sentiment from {self.sentiment_path_in}")

    @staticmethod
    def generate_daily_sentiment(article_df: pd.DataFrame, prices_df: pd.DataFrame,
                                 paths: Paths, s: Settings, schema: Schema, fine_tune: bool) -> pd.DataFrame:
        calendar = TradingCalendar.build_calendar(prices_df, schema)
        article_preprocessor = ArticlePreprocessor(s, schema, calendar)
        ticker_col = schema.article_ticker
        time_col = schema.article_time
        title_col = schema.article_title

        article_df[ticker_col] = article_preprocessor.normalize_tickers(article_df[ticker_col])
        article_df["published_utc"] = ensure_utc(article_df[time_col])
        if s.dedupe_titles:
            article_df = article_df.drop_duplicates(subset=[title_col]).reset_index(drop=True)
        article_df["text"] = article_preprocessor.build_text(article_df)
        article_df = article_df.dropna(subset=[ticker_col, "published_utc"]).reset_index(drop=True)
        article_df = article_df[article_df["text"].str.len() > 0].reset_index(drop=True)
        model = None
        if fine_tune:
            model, tokenizer = FinBertScorer.fine_tune_finbert(article_df, prices_df, schema, s, article_preprocessor)
            scorer = FinBertScorer(s.batch_size, s.max_length, model=model, tokenizer=tokenizer)
        else:
            scorer = FinBertScorer(s.batch_size, s.max_length)
        score_df = scorer.score_texts(article_df["text"].tolist())
        assert len(score_df) == len(article_df), "Score length mismatch."
        article_df = pd.concat([article_df, score_df], axis=1)
        article_df["trading_date"] = article_preprocessor.compute_trading_date(article_df["published_utc"])
        grp = article_df.groupby([ticker_col, "trading_date"], as_index=False).agg(
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
        Logger.info(f"Wrote daily sentiment → {paths.out_sentiment_csv}")
        return grp

