from __future__ import annotations
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
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

class SequenceDataset(Dataset):
    """
    Produces sequences of length lookback with features at each step,
    target is next-day Close (scaled separately).
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, lookback: int):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.lb = lookback
        self.n = len(df)

    def __len__(self):
        return max(0, self.n - self.lb)

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.lb] # (lookback, F)
        y = self.y[idx + self.lb] # scalar target at t+1
        return torch.from_numpy(x), torch.tensor(y)

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
        colmap = {schema.price_close: "Close"}
        if schema.price_open: colmap[schema.price_open] = "Open"
        if schema.price_high: colmap[schema.price_high] = "High"
        if schema.price_low: colmap[schema.price_low] = "Low"
        if schema.price_volume: colmap[schema.price_volume] = "Volume"
        prices = prices.rename(columns=colmap)
        joined = pd.merge(prices, daily_sentiment, on=["ticker", "trading_date"], how="left").sort_values(
            ["ticker", "trading_date"]
        )
        joined["SentimentScore"] = joined["SentimentScore"].fillna(0.0)
        joined["N_t"] = joined["N_t"].fillna(0)
        return joined.reset_index(drop=True)

    @staticmethod
    def build_supervised_for_ticker(df_all: pd.DataFrame, ticker: str, lookback: int, predict_returns: bool) -> Tuple[pd.DataFrame, List[str]]:
        """
        Assemble features (no scaling here) and keep raw Close for later target scaling.
        """
        df = df_all[df_all["ticker"] == ticker].copy().sort_values("trading_date").reset_index(drop=True)
        feat_cols = ["SentimentScore"]
        for c in ["Open", "High", "Low", "Volume", "Close"]:
            if c in df.columns:
                feat_cols.append(c)
        if predict_returns:
            df['raw_target'] = (df['Close'] - df['Open']) / df['Open']
        else:
            df['raw_target'] = df['Close']
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
    def make_dataloaders_for_ticker(df: pd.DataFrame, feat_cols: List[str],
                                    lookback: int, use_arima: bool, batch_size: int):
        """
        Chronological split; fit scalers on TRAIN only; transform all splits; build PyTorch loaders.
        """
        # Splits
        train_all, test = TrainDataPreprocessor._temporal_split(df, train_ratio=0.9)
        train, val = TrainDataPreprocessor._split_train_val(train_all, val_ratio_within_train=0.1)
        # Fit X scaler on TRAIN only
        x_scaler = MinMaxScaler()
        x_scaler.fit(train[feat_cols].astype(float))
        # Transform X across splits
        for part in [train, val, test]:
            part[feat_cols] = part[feat_cols].astype(float)
            part.loc[:, feat_cols] = x_scaler.transform(part[feat_cols])
        # Hybrid ARIMA if enabled
        is_hybrid = use_arima
        arima_pred_train = arima_pred_val = arima_pred_test = None
        if is_hybrid:
            arima_order = (1, 1, 2)
            # Fit on train for train fitted and val forecast
            arima_fit = ARIMA(train['raw_target'], order=arima_order).fit()
            arima_pred_train = arima_fit.fittedvalues.values
            arima_pred_val = arima_fit.forecast(steps=len(val)).values
            # Fit on train_all for test forecast
            arima_fit_full = ARIMA(train_all['raw_target'], order=arima_order).fit()
            arima_pred_test = arima_fit_full.forecast(steps=len(test)).values
            # Set y to residuals
            train['y'] = train['raw_target'] - arima_pred_train
            val['y'] = val['raw_target'] - arima_pred_val
            test['y'] = test['raw_target'] - arima_pred_test
        else:
            train['y'] = train['raw_target']
            val['y'] = val['raw_target']
            test['y'] = test['raw_target']
        # Fit y scaler on TRAIN only
        y_scaler = MinMaxScaler()
        y_scaler.fit(train[['y']])
        # Transform y (residuals or raw target)
        train['y'] = y_scaler.transform(train[['y']]).ravel()
        val['y'] = y_scaler.transform(val[['y']]).ravel()
        test['y'] = y_scaler.transform(test[['y']]).ravel()
        # Datasets
        ds_train = SequenceDataset(train, feat_cols, 'y', lookback)
        ds_val = SequenceDataset(val, feat_cols, 'y', lookback)
        ds_test = SequenceDataset(test, feat_cols, 'y', lookback)
        # Loaders
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)
        # For metrics on original target scale
        original_y_train = train['raw_target'].iloc[lookback:].values
        original_y_val = val['raw_target'].iloc[lookback:].values
        original_y_test = test['raw_target'].iloc[lookback:].values
        if is_hybrid:
            arima_pred_train_slice = arima_pred_train[lookback:]
            arima_pred_val_slice = arima_pred_val[lookback:]
            arima_pred_test_slice = arima_pred_test[lookback:]
        else:
            arima_pred_train_slice = np.zeros_like(original_y_train)
            arima_pred_val_slice = np.zeros_like(original_y_val)
            arima_pred_test_slice = np.zeros_like(original_y_test)
        return (
            dl_train, dl_val, dl_test, y_scaler, is_hybrid,
            arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice,
            original_y_train, original_y_val, original_y_test
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
        use_arima (bool): Whether to include ARIMA predictions.
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
        use_arima_: bool,
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
        self.use_arima = use_arima_
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
                df_joined, ticker=ticker, lookback=self.lookback, predict_returns=self.predict_returns
            )

            # Create DataLoaders
            (dl_train, dl_val, dl_test, y_scaler, is_hybrid,
             arima_pred_train_slice, arima_pred_val_slice, arima_pred_test_slice,
             original_y_train, original_y_val, original_y_test) = \
                TrainDataPreprocessor.make_dataloaders_for_ticker(
                    df_supervised,
                    feat_cols,
                    lookback=self.lookback,
                    use_arima=self.use_arima,
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
