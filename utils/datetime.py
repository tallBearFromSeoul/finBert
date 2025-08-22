from datetime import date, timedelta
from dateutil import parser as dtparser
import pandas as pd
import pytz

def iter_dates(d0: str, d1: str):
    """Yield YYYY-MM-DD strings from d0 to d1 inclusive."""
    cur, end = date.fromisoformat(d0), date.fromisoformat(d1)
    one_day = timedelta(days=1)
    while cur <= end:
        yield cur.isoformat()
        cur += one_day

def _coerce_datetime(x: object, assume_tz: pytz.BaseTzInfo) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        ts = x
    else:
        try:
            ts = pd.Timestamp(dtparser.parse(str(x)))
        except Exception:
            return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize(assume_tz).tz_convert(pytz.UTC)
    return ts.tz_convert(pytz.UTC)

def ensure_utc(series: pd.Series) -> pd.Series:
    tz = pytz.timezone("UTC")
    return series.map(lambda x: _coerce_datetime(x, tz))
