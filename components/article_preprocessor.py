from datetime import date, time
import polars as pl
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

    def build_text(self, df_: pl.DataFrame) -> pl.Series:
        return df_.select(self.get_build_text_expr()).to_series()

    def get_build_text_expr(self) -> pl.Expr:
        title_col = pl.col(self.schema.article_title)
        if self.s.titles_only or self.schema.article_body is None:
            text = title_col.fill_null("").cast(pl.String)
        else:
            body_col = pl.col(self.schema.article_body)
            text = pl.concat_str([
                title_col.fill_null("").cast(pl.String),
                pl.lit(self.s.text_joiner),
                body_col.fill_null("").cast(pl.String)
            ]).str.strip_chars()
        return text.str.replace_all(r"\s+", " ", literal=False)

    def compute_trading_date(self, published_utc_: pl.Series) -> pl.Series:
        temp_df = pl.DataFrame({"published_utc": published_utc_})
        local_ts_expr = pl.col("published_utc").dt.convert_time_zone(self.market_tz.zone)
        after_close_expr = local_ts_expr.dt.time() > pl.lit(self.market_close)
        candidate_date_expr = (
            pl.when(after_close_expr)
            .then(local_ts_expr + pl.duration(days=1))
            .otherwise(local_ts_expr)
            .dt.date()
        )
        mapped_expr = candidate_date_expr.map_elements(lambda d: self._map_to_trading_day(d), return_dtype=pl.Date)
        result = temp_df.select(mapped_expr).to_series()
        return result

    def _map_to_trading_day(self, d_: date) -> date:
        return self.calendar.next_trading_day(d_)

    def get_normalize_tickers_expr(self) -> pl.Expr:
        return pl.col(self.schema.article_ticker).cast(pl.String).str.strip_chars()
