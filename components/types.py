from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd

@dataclass(frozen=True, slots=True)
class Paths:
    fnspid_article_csv: str
    kaggle_article_csv: str
    prices_csv: str
    out_sentiment_csv: str

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
    def __init__(self, trading_days: List[date]):
        if not trading_days:
            raise ValueError("Empty trading_days.")
        self._days = sorted(set(trading_days))

    @property
    def days(self) -> List[date]:
        return self._days

    def next_trading_day(self, d: date) -> date:
        idx = TradingCalendar._bisect_left(self._days, d)
        if idx == len(self._days):
            raise ValueError(f"No trading day on/after {d} in calendar.")
        return self._days[idx]

    @staticmethod
    def _bisect_left(a: List[date], x: date) -> int:
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def build_trading_calendar(prices_df: pd.DataFrame, schema: Schema) -> TradingCalendar:
        dates = pd.to_datetime(prices_df[schema.price_date], errors="coerce").dt.date
        unique_days = dates.dropna().unique().tolist()
        return TradingCalendar(unique_days)
