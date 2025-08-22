from datetime import date, time
import pandas as pd
import pytz

from components.schema import Schema
from components.settings import Settings
from components.trading_calendar import TradingCalendar
from utils.datetime_utils import ensure_utc

class ArticlePreprocessor:
    def __init__(self, settings_: Settings, schema_: Schema,
                 trading_calendar_: TradingCalendar):
        self.s = settings_
        self.schema = schema_
        self.calendar = trading_calendar_
        self.market_tz = pytz.timezone(settings_.market_tz)
        hh, mm = map(int, settings_.market_close_str.split(":"))
        self.market_close = time(hour=hh, minute=mm)

    def build_text(self, df_: pd.DataFrame) -> pd.Series:
        title = df_[self.schema.article_title].fillna("").astype(str)
        if self.s.titles_only or self.schema.article_body is None:
            text = title
        else:
            body = df_[self.schema.article_body].fillna("").astype(str)
            text = (title + self.s.text_joiner + body).str.strip()
        return text.str.replace(r"\s+", " ", regex=True)

    def normalize_time_columns(self, df_: pd.DataFrame) -> pd.Series:
        return ensure_utc(df_[self.schema.article_time])

    def compute_trading_date(self, published_utc_: pd.Series) -> pd.Series:
        local_ts = published_utc_.dt.tz_convert(self.market_tz)
        after_close = local_ts.dt.time > self.market_close
        candidate_date = local_ts.dt.date.where(~after_close, (local_ts + pd.Timedelta(days=1)).dt.date)
        mapped = candidate_date.map(self._map_to_trading_day)
        return pd.to_datetime(mapped)

    def _map_to_trading_day(self, d_: date) -> date:
        return self.calendar.next_trading_day(d_)

    def normalize_tickers(self, df_: pd.DataFrame) -> pd.Series:
        return df_[self.schema.article_ticker].astype(str).str.strip()
