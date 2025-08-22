from dataclasses import dataclass
from pathlib import Path

from utils.logger import Logger

@dataclass(frozen=True, slots=True)
class DataPaths:
    fnspid_article_csv_path: Path
    kaggle_article_csv_path: Path
    prices_csv_dir: Path
    out_sentiment_csv_path: Path

    def __post_init__(self):
        for p in (self.fnspid_article_csv_path, self.kaggle_article_csv_path, self.prices_csv_dir):
            if not p.exists():
                Logger.warning(f"Price file not found: {p}")
