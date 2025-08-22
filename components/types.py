from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional
import pandas as pd

@dataclass(frozen=True, slots=True)
class Paths:
    fnspid_article_csv_path: Path
    kaggle_article_csv_path: Path
    prices_csv_path: Path
    out_sentiment_csv_path: Path

    def __post_init__(self):
        for p in (self.fnspid_article_csv_path, self.kaggle_article_csv_path, self.prices_csv_path):
            if not p.exists():
                raise FileNotFoundError(f"Price file not found: {p}")

@dataclass(frozen=True, slots=True)
class Schema:
    article_ticker: str
    article_time: str
    article_title: str
    article_body: Optional[str]
    price_date: str
    price_open: str
    price_high: str
    price_low: str
    price_close: str
    price_volume: str

@dataclass(frozen=True, slots=True)
class Settings: # Non-frozen for dynamic updates
    market_tz: str
    market_close_str: str
    batch_size: int
    max_length: int # tokenizer truncation
    text_joiner: str = " "
    lower_case_tickers: bool = False
    dedupe_titles: bool = True
    titles_only: bool = True # paper used titles; set False to include body too

class TradingCalendar:
    """
    Trading calendar derived from the prices table.
    Provides next_trading_day(date) by binary search over sorted unique dates.
    """
    def __init__(self, trading_days_: List[date]):
        if not trading_days_:
            raise ValueError("Empty trading_days.")
        self._days = sorted(set(trading_days_))

    @property
    def days(self) -> List[date]:
        return self._days

    def next_trading_day(self, date_: date) -> date:
        idx = TradingCalendar._bisect_left(self._days, date_)
        if idx == len(self._days):
            raise ValueError(f"No trading day on/after {date_} in calendar.")
        return self._days[idx]

    @staticmethod
    def _bisect_left(dates_: List[date], target_date_: date) -> int:
        lo, hi = 0, len(dates_)
        while lo < hi:
            mid = (lo + hi) // 2
            if dates_[mid] < target_date_:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def build_trading_calendar(prices_df_: pd.DataFrame, schema_: Schema) -> TradingCalendar:
        dates = pd.to_datetime(prices_df_[schema_.price_date], errors="coerce").dt.date
        unique_days = dates.dropna().unique().tolist()
        return TradingCalendar(unique_days)
