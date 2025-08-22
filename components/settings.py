from dataclasses import dataclass

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
