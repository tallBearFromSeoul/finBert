from datetime import date, time
import pandas as pd
import pytz

from components.types import Schema, Settings, TradingCalendar
from utils.datetime import ensure_utc

class ArticlePreprocessor:
    def __init__(self, settings: Settings, schema: Schema, calendar: TradingCalendar):
        self.s = settings
        self.schema = schema
        self.calendar = calendar
        self.market_tz = pytz.timezone(settings.market_tz)
        hh, mm = map(int, settings.market_close_str.split(":"))
        self.market_close = time(hour=hh, minute=mm)
    def build_text(self, df: pd.DataFrame) -> pd.Series:
        title = df[self.schema.article_title].fillna("").astype(str)
        if self.s.titles_only or self.schema.article_body is None:
            text = title
        else:
            body = df[self.schema.article_body].fillna("").astype(str)
            text = (title + self.s.text_joiner + body).str.strip()
        return text.str.replace(r"\s+", " ", regex=True)

    def normalize_time_columns(self, df: pd.DataFrame) -> pd.Series:
        return ensure_utc(df[self.schema.article_time])

    def compute_trading_date(self, published_utc: pd.Series) -> pd.Series:
        local_ts = published_utc.dt.tz_convert(self.market_tz)
        after_close = local_ts.dt.time > self.market_close
        candidate_date = local_ts.dt.date.where(~after_close, (local_ts + pd.Timedelta(days=1)).dt.date)
        mapped = candidate_date.map(self._map_to_trading_day)
        return pd.to_datetime(mapped)

    def _map_to_trading_day(self, d: date) -> date:
        return self.calendar.next_trading_day(d)

    def normalize_tickers(self, df: pd.DataFrame) -> pd.Series:
        return df[self.schema.article_ticker].astype(str).str.strip()
