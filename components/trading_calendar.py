from __future__ import annotations
from datetime import date
from typing import List
import polars as pl

from components.schema import Schema

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

    def __len__(self) -> int:
        return len(self.days)

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
    def build_trading_calendar(prices_df_: pl.DataFrame, schema_: Schema) -> TradingCalendar:
        unique_days = prices_df_.select(
            pl.col(schema_.price_date).dt.date().unique().drop_nulls()
        )["date"].to_list()
        return TradingCalendar(unique_days)
