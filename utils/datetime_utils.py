from datetime import date, timedelta
from dateutil import parser as dtparser
import pandas as pd
import pytz

def iter_dates(d0_: str, d1_: str):
    """Yield YYYY-MM-DD strings from d0 to d1 inclusive."""
    cur, end = date.fromisoformat(d0_), date.fromisoformat(d1_)
    one_day = timedelta(days=1)
    while cur <= end:
        yield cur.isoformat()
        cur += one_day

def _coerce_datetime(x_: object, assume_tz_: pytz.BaseTzInfo) -> pd.Timestamp:
    if pd.isna(x_):
        return pd.NaT
    if isinstance(x_, pd.Timestamp):
        ts = x_
    else:
        try:
            ts = pd.Timestamp(dtparser.parse(str(x_)))
        except Exception:
            return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize(assume_tz_).tz_convert(pytz.UTC)
    return ts.tz_convert(pytz.UTC)

def ensure_utc(series_: pd.Series) -> pd.Series:
    tz = pytz.timezone("UTC")
    return series_.map(lambda x: _coerce_datetime(x, tz))
