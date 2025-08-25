from __future__ import annotations
from datetime import datetime
from pathlib import Path
from quantreo.target_engineering.magnitude import _fast_ind_barrier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import torch

from components.schema import Schema
from utils.datetime_utils import ensure_utc
from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

us_holidays = set([
    pd.Timestamp('2010-01-01'), pd.Timestamp('2010-01-18'), pd.Timestamp('2010-02-15'), pd.Timestamp('2010-05-31'),
    pd.Timestamp('2010-07-05'), pd.Timestamp('2010-09-06'), pd.Timestamp('2010-10-11'), pd.Timestamp('2010-11-11'),
    pd.Timestamp('2010-11-25'), pd.Timestamp('2010-12-24'), pd.Timestamp('2010-12-31'), pd.Timestamp('2011-01-17'),
    pd.Timestamp('2011-02-21'), pd.Timestamp('2011-05-30'), pd.Timestamp('2011-07-04'), pd.Timestamp('2011-09-05'),
    pd.Timestamp('2011-10-10'), pd.Timestamp('2011-11-11'), pd.Timestamp('2011-11-24'), pd.Timestamp('2011-12-26'),
    pd.Timestamp('2012-01-02'), pd.Timestamp('2012-01-16'), pd.Timestamp('2012-02-20'), pd.Timestamp('2012-05-28'),
    pd.Timestamp('2012-07-04'), pd.Timestamp('2012-09-03'), pd.Timestamp('2012-10-08'), pd.Timestamp('2012-11-12'),
    pd.Timestamp('2012-11-22'), pd.Timestamp('2012-12-25'), pd.Timestamp('2013-01-01'), pd.Timestamp('2013-01-21'),
    pd.Timestamp('2013-02-18'), pd.Timestamp('2013-05-27'), pd.Timestamp('2013-07-04'), pd.Timestamp('2013-09-02'),
    pd.Timestamp('2013-10-14'), pd.Timestamp('2013-11-11'), pd.Timestamp('2013-11-28'), pd.Timestamp('2013-12-25'),
    pd.Timestamp('2014-01-01'), pd.Timestamp('2014-01-20'), pd.Timestamp('2014-02-17'), pd.Timestamp('2014-05-26'),
    pd.Timestamp('2014-07-04'), pd.Timestamp('2014-09-01'), pd.Timestamp('2014-10-13'), pd.Timestamp('2014-11-11'),
    pd.Timestamp('2014-11-27'), pd.Timestamp('2014-12-25'), pd.Timestamp('2015-01-01'), pd.Timestamp('2015-01-19'),
    pd.Timestamp('2015-02-16'), pd.Timestamp('2015-05-25'), pd.Timestamp('2015-07-03'), pd.Timestamp('2015-09-07'),
    pd.Timestamp('2015-10-12'), pd.Timestamp('2015-11-11'), pd.Timestamp('2015-11-26'), pd.Timestamp('2015-12-25'),
    pd.Timestamp('2016-01-01'), pd.Timestamp('2016-01-18'), pd.Timestamp('2016-02-15'), pd.Timestamp('2016-05-30'),
    pd.Timestamp('2016-07-04'), pd.Timestamp('2016-09-05'), pd.Timestamp('2016-10-10'), pd.Timestamp('2016-11-11'),
    pd.Timestamp('2016-11-24'), pd.Timestamp('2016-12-26'), pd.Timestamp('2017-01-02'), pd.Timestamp('2017-01-16'),
    pd.Timestamp('2017-02-20'), pd.Timestamp('2017-05-29'), pd.Timestamp('2017-07-04'), pd.Timestamp('2017-09-04'),
    pd.Timestamp('2017-10-09'), pd.Timestamp('2017-11-10'), pd.Timestamp('2017-11-23'), pd.Timestamp('2017-12-25'),
    pd.Timestamp('2018-01-01'), pd.Timestamp('2018-01-15'), pd.Timestamp('2018-02-19'), pd.Timestamp('2018-05-28'),
    pd.Timestamp('2018-07-04'), pd.Timestamp('2018-09-03'), pd.Timestamp('2018-10-08'), pd.Timestamp('2018-11-12'),
    pd.Timestamp('2018-11-22'), pd.Timestamp('2018-12-25'), pd.Timestamp('2019-01-01'), pd.Timestamp('2019-01-21'),
    pd.Timestamp('2019-02-18'), pd.Timestamp('2019-05-27'), pd.Timestamp('2019-07-04'), pd.Timestamp('2019-09-02'),
    pd.Timestamp('2019-10-14'), pd.Timestamp('2019-11-11'), pd.Timestamp('2019-11-28'), pd.Timestamp('2019-12-25'),
    pd.Timestamp('2020-01-01')
])

def next_business_day(date):
    date = pd.to_datetime(date) + pd.Timedelta(days=1)  # Always advance to next day first
    while date.weekday() >= 5 or date in us_holidays:
        date += pd.Timedelta(days=1)
    return date.date()

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, valid_indices: List[int], sentiment_col: Optional[str], feature_cols: List[str], target_col: str):
        self.df = df  # full_df, scaled
        self.X = df[feature_cols].values.astype(np.float32)
        self.sent = df[sentiment_col].values.astype(np.float32) if sentiment_col else None
        self.y = df[target_col].values.astype(np.float32)
        self.valid_indices = valid_indices
        print(f"{len(self.valid_indices)=} out of {len(df)=} samples are valid for lookback={lookback} with sentiment_col={sentiment_col}")
        self.lookback = lookback

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        seq_x = self.X[i - self.lookback + 1: i + 1].flatten()
        if self.sent is not None:
            sent_val = self.sent[i]
            x = np.concatenate(([sent_val], seq_x))
        else:
            x = seq_x
        y = self.y[i]
        return torch.from_numpy(x).float(), torch.tensor(y).float()

class TrainDataPreprocessor:
    @staticmethod
    def prepare_joined_frame(daily_sentiment: pd.DataFrame, prices_df: pd.DataFrame, schema: Schema, ticker: str) -> pd.DataFrame:
        prices = prices_df.copy()
        prices["ticker"] = ticker
        prices["trading_date"] = pd.to_datetime(prices[schema.price_date], errors="coerce").dt.date
        keep = ["ticker", "trading_date", schema.price_close]
        for c in [schema.price_open, schema.price_high, schema.price_low, schema.price_volume]:
            if c: keep.append(c)
        prices = prices[keep].dropna(subset=["ticker", "trading_date"])
        colmap = {schema.price_close: "Adj Close"}
        if schema.price_open: colmap[schema.price_open] = "Open"
        if schema.price_high: colmap[schema.price_high] = "High"
        if schema.price_low: colmap[schema.price_low] = "Low"
        if schema.price_volume: colmap[schema.price_volume] = "Volume"
        prices = prices.rename(columns=colmap)
        # Alignment for non-trading days
        ds = daily_sentiment
        ds = ds[ds["ticker"] == ticker].copy()  # Filter to this ticker only
        ds['next_date'] = ds['trading_date'].apply(next_business_day)
        ds["trading_date"] = pd.to_datetime(ds["trading_date"], errors="coerce").dt.date
        # Diagnostic prints
        joined = pd.merge(prices, ds, on=["ticker", "trading_date"], how="left").sort_values(
            ["ticker", "trading_date"]
        )
        return joined.reset_index(drop=True)

    @staticmethod
    def build_supervised_for_ticker(df_all: pd.DataFrame, ticker: str, predict_returns: bool) -> Tuple[pd.DataFrame, List[str]]:
        df = df_all[df_all["ticker"] == ticker].copy().sort_values("trading_date").reset_index(drop=True)
        feat_cols = ["Open", "High", "Low", "Volume", "Adj Close"]
        if predict_returns:
            df['raw_target'] = (df['Adj Close'].shift(-1) - df['Adj Close']) / df['Adj Close']
        else:
            df['raw_target'] = df['Adj Close'].shift(-1)
        df = df.dropna(subset=['raw_target'])
        return df, feat_cols

    @staticmethod
    def _temporal_split(df: pd.DataFrame, train_ratio: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n = len(df)
        cut = int(math.floor(n * train_ratio))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    @staticmethod
    def _split_train_val(train_df: pd.DataFrame,
                         val_ratio_within_train: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n = len(train_df)
        cut = int(math.floor(n * (1 - val_ratio_within_train)))
        return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()

    @staticmethod
    def make_dataloaders_for_ticker(df: pd.DataFrame, sentiment_col: Optional[str], feat_cols: List[str],
                                    lookback: int, batch_size: int, scale_method: str):
        scaler_cls = MinMaxScaler if scale_method == "minmax" else StandardScaler
        n = len(df)
        has_sent = df[sentiment_col].notna() if sentiment_col else pd.Series([True] * n, index=df.index)
        valid_indices = [i for i in range(n) if has_sent[i] and i >= lookback - 1]
        num_valid = len(valid_indices)
        if num_valid == 0:
            raise ValueError("No valid samples found")
        train_ratio = 0.9
        train_all_valid_num = int(math.floor(num_valid * train_ratio))
        val_ratio_within_train = 0.1
        train_valid_num = int(math.floor(train_all_valid_num * (1 - val_ratio_within_train)))
        valid_train = valid_indices[:train_valid_num]
        valid_val = valid_indices[train_valid_num:train_all_valid_num]
        valid_test = valid_indices[train_all_valid_num:]
        if valid_train:
            train_start_row = max(0, min(i - lookback + 1 for i in valid_train))
            train_end_row = max(valid_train) + 1
        else:
            train_start_row = 0
            train_end_row = 0
        train_feat_df = df.iloc[train_start_row:train_end_row]
        x_scaler = scaler_cls()
        x_scaler.fit(train_feat_df[feat_cols].astype(float))
        df[feat_cols] = x_scaler.transform(df[feat_cols].astype(float))
        df['y'] = df['raw_target']
        if valid_train:
            y_train = df['raw_target'].iloc[valid_train].values.reshape(-1, 1)
            y_scaler = scaler_cls()
            y_scaler.fit(y_train)
            df['y'] = y_scaler.transform(df[['y']]).ravel()
        else:
            raise ValueError("No train samples")
        ds_train = SequenceDataset(df, lookback, valid_train, sentiment_col, feat_cols, 'y')
        ds_val = SequenceDataset(df, lookback, valid_val, sentiment_col, feat_cols, 'y')
        ds_test = SequenceDataset(df, lookback, valid_test, sentiment_col, feat_cols, 'y')
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
        original_y_train = df['raw_target'].iloc[valid_train].values if valid_train else np.array([])
        original_y_val = df['raw_target'].iloc[valid_val].values if valid_val else np.array([])
        original_y_test = df['raw_target'].iloc[valid_test].values if valid_test else np.array([])
        dates_train = df['trading_date'].iloc[valid_train].values if valid_train else np.array([])
        dates_val = df['trading_date'].iloc[valid_val].values if valid_val else np.array([])
        dates_test = df['trading_date'].iloc[valid_test].values if valid_test else np.array([])
        return (
            dl_train, dl_val, dl_test, y_scaler,
            original_y_train, original_y_val, original_y_test,
            dates_train, dates_val, dates_test
        )

class TrainDataLoader(Iterator):
    """
    A wrapper class for lazily creating and yielding PyTorch DataLoaders for stock price data.

    Attributes:
        prices_dir (Path): Directory containing price CSV files.
        tickers (List[str]): List of stock tickers to process.
        daily_sentiment (pd.DataFrame): Sentiment data DataFrame.
        schema (Schema): Schema for data preprocessing.
        start_date (datetime): Start date for filtering data.
        end_date (datetime): End date for filtering data.
        lookback (int): Lookback period for supervised learning.
        predict_returns (bool): Whether to predict returns instead of prices.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to pin memory for GPU transfer.
        seed (Optional[int]): Random seed for reproducibility.
        logger (logging.Logger): Logger for tracking operations.
    """

    def __init__(
        self,
        prices_dir_: Path,
        tickers_: List[str],
        daily_sentiment_: pd.DataFrame,
        schema_: Schema,
        start_date_: datetime,
        end_date_: datetime,
        lookback_: int,
        batch_size_: int,
        predict_returns_: bool,
        num_workers_: int = 0,
        pin_memory_: bool = True,
        seed_: Optional[int] = None,
    ) -> None:
        self.prices_dir = prices_dir_
        self.tickers = tickers_
        self.daily_sentiment = daily_sentiment_
        self.schema = schema_
        self.start_date = start_date_
        self.end_date = end_date_
        self.lookback = lookback_
        self.predict_returns = predict_returns_
        self.batch_size = batch_size_
        self.num_workers = num_workers_
        self.pin_memory = pin_memory_
        self.seed = seed_

        # Validate inputs
        self._validate_inputs()

        # Set random seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Iterator state
        self._ticker_index = 0

    def _validate_inputs(self) -> None:
        """Validate initialization parameters."""
        if not self.prices_dir.is_dir():
            Logger.error(f"Prices directory does not exist: {self.prices_dir}")
            raise ValueError(f"Prices directory does not exist")
        if not isinstance(self.tickers, list) or not all(isinstance(t, str) for t in self.tickers):
            Logger.error("Tickers must be a list of strings")
            raise ValueError("Tickers must be a list of strings")
        if not isinstance(self.daily_sentiment, pd.DataFrame):
            Logger.error("daily_sentiment must be a pandas DataFrame")
            raise TypeError("daily_sentiment must be a pandas DataFrame")
        if self.batch_size <= 0:
            Logger.error("Batch size must be a positive integer")
            raise ValueError("Batch size must be a positive integer")
        if self.num_workers < 0:
            Logger.error("Number of workers must be non-negative")
            raise ValueError("Number of workers must be non-negative")
        if self.lookback <= 0:
            Logger.error("Lookback must be a positive integer")
            raise ValueError("Lookback must be a positive integer")

    def _process_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Process a single ticker and create its DataLoaders.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with DataLoaders and metadata, or None if skipped.
        """
        try:
            Logger.info(f"Processing ticker: {ticker}")
            price_path = self.prices_dir / f"{ticker}.csv"
            if not price_path.exists():
                Logger.warning(f"Skipping {ticker}: price file not found")
                return None

            # Load and preprocess price data
            prices_df = pd.read_csv(price_path, parse_dates=["date"], date_format="%Y-%m-%d")
            prices_df["date"] = ensure_utc(prices_df["date"])
            prices_df = prices_df[
                (prices_df["date"] > self.start_date) &
                (prices_df["date"] < self.end_date)
            ].dropna(subset=["date"])
            Logger.info(f"Loaded prices for {ticker} shape={prices_df.shape}")

            # Join sentiment and price data
            df_joined = TrainDataPreprocessor.prepare_joined_frame(
                self.daily_sentiment, prices_df, self.schema, ticker
            )

            # Build supervised dataset
            df_supervised, feat_cols = TrainDataPreprocessor.build_supervised_for_ticker(
                df_joined, ticker=ticker, predict_returns=self.predict_returns
            )

            # Create DataLoaders
            (dl_train, dl_val, dl_test, y_scaler, is_hybrid,
             arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice,
             original_y_train, original_y_val, original_y_test) = \
                TrainDataPreprocessor.make_dataloaders_for_ticker(
                    df_supervised,
                    feat_cols,
                    lookback=self.lookback,
                    batch_size=self.batch_size)

            # Configure DataLoaders
            dl_train = self._configure_dataloader(dl_train.dataset)
            dl_val = self._configure_dataloader(dl_val.dataset)
            dl_test = self._configure_dataloader(dl_test.dataset)

            # Store results
            data = {
                "train": dl_train,
                "val": dl_val,
                "test": dl_test,
                "y_scaler": y_scaler,
                "is_hybrid": is_hybrid,
                "arima_pred_train": arima_pred_train_slice,
                "arima_pred_val": arima_pred_val_slice,
                "arima_pred_test": arima_pred_test_slice,
                "original_y_train": original_y_train,
                "original_y_val": original_y_val,
                "original_y_test": original_y_test,
                "feature_columns": feat_cols
            }

            Logger.info(f"DataLoaders created for {ticker}")
            return data

        except Exception as e:
            Logger.error(f"Failed to process {ticker}: {str(e)}")
            return None

    def _configure_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Configure a DataLoader with the wrapper's settings.

        Args:
            dataset (Dataset): The dataset to wrap.

        Returns:
            DataLoader: Configured DataLoader instance.
        """
        try:
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        except Exception as e:
            Logger.error(f"Failed to configure DataLoader: {str(e)}")
            raise RuntimeError(f"Failed to configure DataLoader: {str(e)}")

    def __iter__(self) -> TrainDataLoader:
        """Return the iterator object."""
        self._ticker_index = 0
        return self

    def __next__(self) -> Tuple[str, Dict[str, Any]]:
        """Fetch the next ticker's data lazily."""
        while self._ticker_index < len(self.tickers):
            ticker = self.tickers[self._ticker_index]
            self._ticker_index += 1
            data = self._process_ticker(ticker)
            if data is not None:
                return ticker, data
        raise StopIteration

    def __len__(self) -> int:
        """Return the number of tickers."""
        return len(self.tickers)
