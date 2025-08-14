from dataclasses import dataclass
from typing import Optional

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

