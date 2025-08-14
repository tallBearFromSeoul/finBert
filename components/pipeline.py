from __future__ import annotations
from datetime import datetime
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from typing import Dict, List, Optional, Tuple, Union
from yfinance import Ticker
import argparse
import concurrent.futures
import dask.dataframe as dd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
import os
import torch

from components.sentiment_generator import SentimentGenerator
from components.data_paths import DataPaths
from components.schema import Schema
from components.settings import Settings
from components.train_data_loader import TrainDataPreprocessor
from components.train import evaluate, train_model
from utils.datetime_utils import ensure_utc
from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

class Pipeline:
    def __init__(self):
        args, runtime = Pipeline.argparse()
        if args.seed is not None:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            Logger.info(f"Set deterministic seeds to {args.seed}")
        # Expand/normalize user paths
        prices_dir = Path(os.path.expanduser(args.prices_dir)).resolve()
        tickers_path = Path(os.path.expanduser(args.tickers_path)).resolve()
        fnspid_csv_path = Path(os.path.expanduser(args.fnspid_csv_path)).resolve()
        kaggle_csv_path = Path(os.path.expanduser(args.kaggle_csv_path)).resolve()
        sentiment_csv_path_in = Path(os.path.expanduser(args.sentiment_csv_path_in)).resolve() \
            if args.sentiment_csv_path_in else None
        out_root = ensure_dir(Path("output") / runtime)
        out_logs = ensure_dir(out_root / "logs")
        Logger.setup_file(out_logs / "pipeline.log")
        with open(out_root / "run_args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        load_dir = Path(os.path.expanduser(args.load_dir)).resolve() if args.load_dir else None
        # Output layout
        saved_root = ensure_dir(out_root / "saved_weights")
        eval_json_path = out_root / "evaluation.json"
        # Settings for paper alignment
        settings = Settings(
            market_tz=args.market_tz,
            market_close_str=args.market_close,
            batch_size=args.batch_size,
            max_length=args.max_length,
            titles_only=not args.use_bodies,
            dedupe_titles=True,
        )
        start_date = pd.Timestamp(2010, 1, 1, tz='UTC') # Year, Month, Day, UTC timezone
        end_date = pd.Timestamp(2020, 1, 1, tz='UTC') # Align to paper's "up to 2019"
        # Load and filter the news DataFrame (once)
        Logger.info(f"Loading and filtering news data from {args.data_source}...")
        article_df = Pipeline.load_and_filter_news(fnspid_csv_path, kaggle_csv_path, args.data_source, start_date, end_date, args.use_bodies)
        Logger.info(f"Loaded news dataframe with columns={list(article_df.columns)}.")
        # Schema
        schema = Schema(
            article_ticker="stock",
            article_time="date",
            article_title="title",
            article_body="body" if args.use_bodies else None,
            price_date="date",
            price_open="open",
            price_high="high",
            price_low="low",
            price_close="close",
            price_volume="volume",
        )
        assert isinstance(args.ticker, list), "Sanity check on args.ticker type"
        # Determine tickers to run
        if args.ticker[0].lower() == "all-tickers":
            tickers = Pipeline._list_all_tickers(Path(tickers_path))
            if not tickers:
                raise ValueError(f"No price CSVs found under {tickers_path}")
            Logger.info(f"Running all tickers: {len(tickers)} found.")
        else:
            tickers = args.ticker
        out_sentiment_csv = out_root / "sentiment_daily.csv"
        data_paths = DataPaths(fnspid_csv_path, kaggle_csv_path, prices_dir, out_sentiment_csv)
        sentiment_generator = SentimentGenerator(
            tickers, data_paths, settings, schema, sentiment_csv_path_in, article_df,
            start_date, end_date, (out_root / args.fine_tune_dir), args.fine_tune_load_path)
        if args.train:
            self.train(tickers, sentiment_generator.daily_sentiment.to_pandas(), prices_dir, schema,
                       start_date, end_date, args, out_root, saved_root, eval_json_path, load_dir)

    @staticmethod
    def train(
        tickers_: List[str], daily_sentiment_: pd.DataFrame, prices_dir_: Path,
        schema_: Schema, start_date_: datetime, end_date_: datetime, args_: argparse.Namespace,
        out_root_: Path, saved_root_: Path, eval_path_: Path, load_dir_: Optional[Path]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, Optional[List[str]]]:
        eval_rows: List[Dict[str, object]] = []
        df_supervised_list = []
        df_joined_list = []
        ticker_lengths = []
        ticker_joined_lengths = []
        Logger.info(f"Training predictor for ticker: {tickers_}")
        def process_ticker(ticker):
            prices_path = prices_dir_ / f"{ticker}.csv"
            if not prices_path.exists():
                Logger.info(f"Skipping {ticker}: price file not found ({prices_path}), downloading...")
                ticker_data = Ticker(ticker)
                prices_df = ticker_data.history(start='2005-10-14', end='2023-12-29')
                prices_df.reset_index(inplace=True)
                prices_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                          'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                prices_df["date"] = ensure_utc(prices_df["date"])
                prices_df = prices_df[(prices_df["date"] > start_date_) & (prices_df["date"] < end_date_)].dropna(subset=["date"])
                prices_df.to_csv(str(prices_path), index=False)
            else:
                prices_df = pd.read_csv(prices_path, parse_dates=["date"], date_format="%Y-%m-%d")

            if (prices_df[['open', 'high', 'low', 'close']].max() > 1e4).any():
                Logger.warning(f"Skipping {ticker}: higher price values than 1e5 found in {prices_path}")
                return pd.DataFrame(), pd.DataFrame(), 0, 0, None

            if prices_df.empty:
                Logger.warning(f"Skipping {ticker}: price file is empty after loading ({prices_path})")
                return pd.DataFrame(), pd.DataFrame(), 0, 0, None

            assert (prices_df[['open', 'high', 'low', 'close']].max() <= 1e4).all(), f"High price values found in {prices_path}"

            Logger.info(f"Loaded prices for {ticker} with columns={list(prices_df.columns)} shape={prices_df.shape}.")
            # Join
            df_joined = TrainDataPreprocessor.prepare_joined_frame(daily_sentiment_, prices_df, schema_, ticker)
            df_supervised, feat_cols = TrainDataPreprocessor.build_supervised_for_ticker(
                df_joined, ticker, args_.predict_returns
            )
            return df_supervised, df_joined, len(df_supervised), len(df_joined), feat_cols

        feat_cols = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_ticker, ticker) for ticker in tickers_]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                df_supervised_list.append(result[0])
                df_joined_list.append(result[1])
                ticker_lengths.append(result[2])
                ticker_joined_lengths.append(result[3])
                if result[4] is not None:
                    if feat_cols is not None:
                        assert feat_cols == result[4], "Feature columns mismatch between tickers"
                    feat_cols = result[4]
        Logger.info(f"{feat_cols=}")
        df_supervised = pd.concat(df_supervised_list, ignore_index=True)
        df_joined = pd.concat(df_joined_list, ignore_index=True)
        Logger.info(f"Combined supervised dataframe shape: {df_supervised.shape}")
        sentiment_col = 'SentimentScore'
        use_sentiment_in_features = 'finbert' in args_.model.lower()
        (dl_train, dl_val, dl_test, y_scaler,
         original_y_train, original_y_val, original_y_test,
         dates_train, dates_val, dates_test) = TrainDataPreprocessor.make_dataloaders_for_ticker(
            df_supervised, sentiment_col, use_sentiment_in_features,
            feat_cols, args_.lookback, args_.batch_size, args_.scale_method, args_.sentiment_threshold
        )
        ticker = "all_tickers"
        # Prepare saved weights dir for this ticker
        ticker_save_dir = ensure_dir(saved_root_ / ticker)
        # Resolve initial load path if provided
        resolved_load_path = None
        if load_dir_ is not None:
            cand = Pipeline.find_weight_file_for_ticker(load_dir_, ticker)
            if cand:
                resolved_load_path = cand
                Logger.info(f"--load-dir provided, found weights {ticker} {cand} in {load_dir_}")
            else:
                Logger.warning(f"--load-dir provided, but no weights found for {ticker} in {load_dir_}")
        input_size = len(feat_cols) * args_.lookback + (1 if use_sentiment_in_features else 0)
        model_type = args_.model.split('-')[-1]
        # Train (now passes y_scaler, returns both scaled and price metrics)
        model, train_hist, val_hist, best_info, price_hist = train_model(
            dl_train, dl_val, input_size=input_size,
            original_y_train=original_y_train,
            original_y_val=original_y_val,
            model_type=model_type,
            epochs=args_.epochs, patience=args_.patience,
            lr=args_.lr, weight_decay=args_.weight_decay, dropout_rate=args_.dropout_rate,
            save_dir=ticker_save_dir,
            ticker=ticker,
            load_path=resolved_load_path,
            y_scaler=y_scaler
        )
        # Evaluate
        metrics_train = evaluate(model, dl_train, y_scaler)
        metrics_val = evaluate(model, dl_val, y_scaler)
        metrics_test = evaluate(model, dl_test, y_scaler)
        metrics = metrics_test
        lb = args_.lookback
        y_true_train = metrics_train['y_true']
        y_pred_train = metrics_train['y_pred']
        y_true_val = metrics_val['y_true']
        y_pred_val = metrics_val['y_pred']
        y_true_test = metrics_test['y_true']
        y_pred_test = metrics_test['y_pred']
        # Determine target type for prediction plot
        if args_.predict_returns:
            target_type = 'returns'
        else:
            target_type = 'price'
        # Visualize predictions
        dates_full = np.arange(len(dates_train) + len(dates_val) + len(dates_test))
        y_true_full = np.concatenate([y_true_train, y_true_val, y_true_test])
        y_pred_full = np.concatenate([y_pred_train, y_pred_val, y_pred_test])
        train_end_idx = len(y_true_train)
        val_end_idx = train_end_idx + len(y_true_val)
        predictions: Dict[str, Dict[str, np.ndarray]] = {'dates': dates_full,
                                                         'y_true': y_true_full,
                                                         'y_pred': y_pred_full}
        val_start_date = dates_full[train_end_idx] if train_end_idx < len(dates_full) else dates_full[-1]
        test_start_date = dates_full[val_end_idx] if val_end_idx < len(dates_full) else dates_full[-1]
        Pipeline.visualize_full_prediction(
            predictions, target_type, ticker, args_.model, out_root_,
            val_start_date, test_start_date, tickers_
        )
        # Visualize sentiment (full series)
        sentiment_scores = df_joined['SentimentScore'].values
        full_dates = df_joined['trading_date'].values
        Pipeline.visualize_sentiment(sentiment_scores, full_dates, ticker, out_root_, tickers_, ticker_joined_lengths, args_.sentiment_threshold)
        Logger.info(
            f"[{ticker}] Test (price) MSE={metrics['test_MSE']:.9f} | MAE={metrics['test_MAE']:.9f} | RMSE={metrics['test_RMSE']:.9f} "
            f"(scaled MSE={metrics['test_MSE_scaled']:.9f})"
        )
        if len(train_hist) and len(val_hist):
            Logger.info(
                f"[{ticker}] Final (price) Train MSE={price_hist['train_mse'][-1]:.9f} | "
                f"Val MSE={price_hist['val_mse'][-1]:.9f} "
                f"(scaled Train MSE={train_hist[-1]:.9f} | Val MSE={val_hist[-1]:.9f})"
            )
        Logger.info(
            f"[{ticker}] Best epoch={best_info['best_epoch']} | "
            f"Best (price) Train MSE={best_info['best_train_mse']:.9f} | "
            f"Best (price) Val MSE={best_info['best_val_mse']:.9f} | Saved at={best_info['best_path']}"
        )
        curve_data = {
            "epoch": np.arange(1, len(train_hist) + 1, dtype=int).tolist(),
            # scaled losses (training objective)
            "train_mse_scaled": train_hist,
            "val_mse_scaled": val_hist,
            # price-scale metrics (comparable to test)
            "train_mse_price": price_hist["train_mse"],
            "val_mse_price": price_hist["val_mse"],
            "train_mae_price": price_hist["train_mae"],
            "val_mae_price": price_hist["val_mae"],
            "train_rmse_price": price_hist["train_rmse"],
            "val_rmse_price": price_hist["val_rmse"],
        }
        with open(out_root_ / f"{ticker}_{args_.model}_training_curve.json", 'w') as f: # CHANGED: Include model in filename
            json.dump(curve_data, f, indent=4)
        # Evaluation rows (consistent price-scale metrics, plus scaled for reference)
        eval_rows.append({
            "ticker": ticker,
            "best_epoch": best_info["best_epoch"],
            # canonical (price-scale)
            "best_train_mse": best_info["best_train_mse"],
            "best_val_mse": best_info["best_val_mse"],
            "best_train_mae": best_info["best_train_mae"],
            "best_val_mae": best_info["best_val_mae"],
            "best_train_rmse": best_info["best_train_rmse"],
            "best_val_rmse": best_info["best_val_rmse"],
            "saved_weights_path": best_info["best_path"],
            "test_MSE": metrics["test_MSE"],
            "test_MAE": metrics["test_MAE"],
            "test_RMSE": metrics["test_RMSE"],
            # scaled (for reference / debugging)
            "best_train_mse_scaled": best_info["best_train_mse_scaled"],
            "best_val_mse_scaled": best_info["best_val_mse_scaled"],
            "test_MSE_scaled": metrics["test_MSE_scaled"],
            "test_MAE_scaled": metrics["test_MAE_scaled"],
            "test_RMSE_scaled": metrics["test_RMSE_scaled"],
            "epochs_run": best_info["epochs_run"],
            "final_train_mse": best_info["final_train_mse"], # price
            "final_val_mse": best_info["final_val_mse"], # price
            "final_train_mse_scaled": best_info["final_train_mse_scaled"],
            "final_val_mse_scaled": best_info["final_val_mse_scaled"],
        })
        if eval_rows:
            with open(eval_path_, 'w') as f:
                json.dump(eval_rows, f, indent=4)
            Logger.info(f"Wrote evaluation summary → {eval_path_}")
        else:
            Logger.warning("No evaluation rows to write; did all tickers_ fail to run?")

    @staticmethod
    def visualize_sentiment(sentiment_scores_: np.ndarray, dates_: np.ndarray,
                            ticker_: str, out_root_: Path, tickers_: List[str],
                            ticker_lengths_: List[int], sentiment_threshold_: float):
        """
        Visualize daily sentiment scores over time and save as PNG.
        """
        plot_dir = ensure_dir(out_root_ / "plots")
        fig, ax = plt.subplots(figsize=(20, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(tickers_)))
        cum_row = 0
        for i, tick in enumerate(tickers_):
            length = ticker_lengths_[i]
            end_row = cum_row + length
            start_j = cum_row
            end_j = end_row
            if start_j < end_j:
                color = colors[i]
                scores_slice = sentiment_scores_[start_j:end_j]
                dates_slice = dates_[start_j:end_j]
                mask = ~np.isnan(scores_slice) & (np.abs(scores_slice) > sentiment_threshold_)
                ax.scatter(dates_slice[mask], scores_slice[mask], color=color, label=f'Sentiment {tick}', s=20)
            cum_row = end_row
        ax.set_title(f'{ticker_} Daily Sentiment')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True)
        plt.gcf().autofmt_xdate()
        plt.savefig(plot_dir / f"{ticker_}_sentiment.png")
        plt.close()
        Logger.info(f"Saved sentiment plot for {ticker_} to {plot_dir / f'{ticker_}_sentiment.png'}")

    @staticmethod
    def visualize_full_prediction(predictions_: Dict[str, np.ndarray],
                                target_type_: str, ticker_: str, model_: str, out_root_: Path,
                                val_start_date_, test_start_date_, tickers_: List[str]):
        """
        Visualize stitched true vs predicted values for train+val+test and save as PNG.
        Extended to include comprehensive regression diagnostic plots in subplots.
        Differentiates train, val, test in all plots with color coding and separate diagnostics.
        """
        plot_dir = ensure_dir(out_root_ / "plots")
        # Main figure with gridspec for multiple subplots
        fig = plt.figure(figsize=(22, 40))
        gs = gridspec.GridSpec(11, 2, figure=fig)

        d = predictions_
        dates = d['dates']
        y_true = d['y_true']
        y_pred = d['y_pred']
        residuals = y_true - y_pred

        # Ensure date compatibility
        if len(dates) > 0 and isinstance(dates[0], np.datetime64):
            val_start_date_ = np.datetime64(val_start_date_)
            test_start_date_ = np.datetime64(test_start_date_)

        # Create masks for train, val, test
        train_mask = dates < val_start_date_
        val_mask = (dates >= val_start_date_) & (dates < test_start_date_)
        test_mask = dates >= test_start_date_

        # Indices for length checks
        train_idx = np.nonzero(train_mask)[0]
        val_idx = np.nonzero(val_mask)[0]
        test_idx = np.nonzero(test_mask)[0]

        # Color definitions
        split_colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
        true_colors = {'train': 'darkblue', 'val': 'darkgreen', 'test': 'darkred'}
        pred_colors = {'train': 'blue', 'val': 'green', 'test': 'red'}

        # Row 0: Original time series plot (spans both columns)
        ax_time = fig.add_subplot(gs[0, :])
        for split in ['train', 'val', 'test']:
            mask = locals()[f'{split}_mask']
            ax_time.scatter(dates[mask], y_true[mask], color=true_colors[split], label=f'True {split.capitalize()}', marker='o', s=10, alpha=0.6)
            ax_time.scatter(dates[mask], y_pred[mask], color=pred_colors[split], label=f'Pred {split.capitalize()}', marker='x', s=10, alpha=0.6)
        for date, true, pred in zip(dates, y_true, y_pred):
            ax_time.vlines(x=date, ymin=min(true, pred), ymax=max(true, pred), color='black', linestyle='-', alpha=0.3)
        ax_time.axvline(val_start_date_, color='green', linestyle='--', label='Start of Validation')
        ax_time.axvline(test_start_date_, color='red', linestyle='--', label='Start of Test')
        ax_time.set_title(f'True vs Predicted over Time')
        ax_time.set_xlabel('Date')
        ylabel = target_type_.capitalize()
        ax_time.set_ylabel(ylabel)
        ax_time.legend(loc='upper left', ncol=2)
        ax_time.grid(True)

        # Row 1, Col 0: Actual vs Predicted scatter
        ax_act_pred = fig.add_subplot(gs[1, 0])
        for split in ['train', 'val', 'test']:
            mask = locals()[f'{split}_mask']
            ax_act_pred.scatter(y_true[mask], y_pred[mask], color=split_colors[split], alpha=0.5, label=split.capitalize())
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax_act_pred.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y = x')
        ax_act_pred.set_title('Actual vs Predicted')
        ax_act_pred.set_xlabel('Actual')
        ax_act_pred.set_ylabel('Predicted')
        ax_act_pred.legend()
        ax_act_pred.grid(True)

        # Row 1, Col 1: Residuals vs Predicted
        ax_res_pred = fig.add_subplot(gs[1, 1])
        for split in ['train', 'val', 'test']:
            mask = locals()[f'{split}_mask']
            ax_res_pred.scatter(y_pred[mask], residuals[mask], color=split_colors[split], alpha=0.5, label=split.capitalize())
        ax_res_pred.axhline(0, color='black', linestyle='--')
        ax_res_pred.set_title('Residuals vs Predicted')
        ax_res_pred.set_xlabel('Predicted')
        ax_res_pred.set_ylabel('Residuals')
        ax_res_pred.legend()
        ax_res_pred.grid(True)

        # Row 2: Residuals over Time (spans both columns)
        ax_res_time = fig.add_subplot(gs[2, :])
        for split in ['train', 'val', 'test']:
            mask = locals()[f'{split}_mask']
            ax_res_time.scatter(dates[mask], residuals[mask], color=split_colors[split], alpha=0.5, label=split.capitalize())
        ax_res_time.axhline(0, color='black', linestyle='--')
        ax_res_time.set_title('Residuals over Time')
        ax_res_time.set_xlabel('Date')
        ax_res_time.set_ylabel('Residuals')
        ax_res_time.legend()
        ax_res_time.grid(True)

        # Overall diagnostics
        # Row 3, Col 0: QQ Plot All
        ax_qq_all = fig.add_subplot(gs[3, 0])
        if len(residuals) > 0:
            qqplot(residuals, line='s', ax=ax_qq_all)
        ax_qq_all.set_title('QQ Plot of All Residuals')

        # Row 3, Col 1: Histogram of All Residuals
        ax_hist_all = fig.add_subplot(gs[3, 1])
        if len(residuals) > 0:
            num_bins_all = 100
            ax_hist_all.hist(residuals, bins=num_bins_all, color='purple', alpha=0.7)
        ax_hist_all.set_title('Histogram of All Residuals')
        ax_hist_all.set_xlabel('Residuals')
        ax_hist_all.set_ylabel('Frequency')
        ax_hist_all.grid(True)

        # Row 4: ACF of All Residuals (spans both columns)
        ax_acf_all = fig.add_subplot(gs[4, :])
        if len(residuals) > 1:
            plot_acf(residuals, lags=min(40, len(residuals)-1), ax=ax_acf_all)
        ax_acf_all.set_title('Autocorrelation of All Residuals')

        # Train diagnostics
        # Row 5, Col 0: QQ Plot Train
        ax_qq_train = fig.add_subplot(gs[5, 0])
        if len(train_idx) > 0:
            qqplot(residuals[train_mask], line='s', ax=ax_qq_train)
        ax_qq_train.set_title('QQ Plot of Train Residuals')

        # Row 5, Col 1: Histogram of Train Residuals
        ax_hist_train = fig.add_subplot(gs[5, 1])
        if len(train_idx) > 0:
            num_bins_train = 100
            ax_hist_train.hist(residuals[train_mask], bins=num_bins_train, color='blue', alpha=0.7)
        ax_hist_train.set_title('Histogram of Train Residuals')
        ax_hist_train.set_xlabel('Residuals')
        ax_hist_train.set_ylabel('Frequency')
        ax_hist_train.grid(True)

        # Row 6: ACF of Train Residuals (spans both columns)
        ax_acf_train = fig.add_subplot(gs[6, :])
        if len(train_idx) > 1:
            plot_acf(residuals[train_mask], lags=min(40, len(train_idx)-1), ax=ax_acf_train)
        ax_acf_train.set_title('Autocorrelation of Train Residuals')

        # Val diagnostics
        # Row 7, Col 0: QQ Plot Val
        ax_qq_val = fig.add_subplot(gs[7, 0])
        if len(val_idx) > 0:
            qqplot(residuals[val_mask], line='s', ax=ax_qq_val)
        ax_qq_val.set_title('QQ Plot of Val Residuals')

        # Row 7, Col 1: Histogram of Val Residuals
        ax_hist_val = fig.add_subplot(gs[7, 1])
        if len(val_idx) > 0:
            num_bins_val = 100
            ax_hist_val.hist(residuals[val_mask], bins=num_bins_val, color='green', alpha=0.7)
        ax_hist_val.set_title('Histogram of Val Residuals')
        ax_hist_val.set_xlabel('Residuals')
        ax_hist_val.set_ylabel('Frequency')
        ax_hist_val.grid(True)

        # Row 8: ACF of Val Residuals (spans both columns)
        ax_acf_val = fig.add_subplot(gs[8, :])
        if len(val_idx) > 1:
            plot_acf(residuals[val_mask], lags=min(40, len(val_idx)-1), ax=ax_acf_val)
        ax_acf_val.set_title('Autocorrelation of Val Residuals')

        # Test diagnostics
        # Row 9, Col 0: QQ Plot Test
        ax_qq_test = fig.add_subplot(gs[9, 0])
        if len(test_idx) > 0:
            qqplot(residuals[test_mask], line='s', ax=ax_qq_test)
        ax_qq_test.set_title('QQ Plot of Test Residuals')

        # Row 9, Col 1: Histogram of Test Residuals
        ax_hist_test = fig.add_subplot(gs[9, 1])
        if len(test_idx) > 0:
            num_bins_test = 100
            ax_hist_test.hist(residuals[test_mask], bins=num_bins_test, color='red', alpha=0.7)
        ax_hist_test.set_title('Histogram of Test Residuals')
        ax_hist_test.set_xlabel('Residuals')
        ax_hist_test.set_ylabel('Frequency')
        ax_hist_test.grid(True)

        # Row 10: ACF of Test Residuals (spans both columns)
        ax_acf_test = fig.add_subplot(gs[10, :])
        if len(test_idx) > 1:
            plot_acf(residuals[test_mask], lags=min(40, len(test_idx)-1), ax=ax_acf_test)
        ax_acf_test.set_title('Autocorrelation of Test Residuals')

        # Overall figure title and adjustments
        fig.suptitle(f'{tickers_} {model_.upper()} Full {target_type_.capitalize()} Prediction and Diagnostics', fontsize=16)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(plot_dir / f"{ticker_}_{model_}_full_{target_type_}_prediction.png")
        plt.close()
        Logger.info(f"Saved full prediction plot for {tickers_} to {plot_dir / f'{ticker_}_{model_}_full_{target_type_}_prediction.png'}")

    @staticmethod
    def find_weight_file_for_ticker_(load_dir_: Path, ticker_: str) -> Optional[Path]:
        if load_dir_.is_file():
            return load_dir_
        if not load_dir_.exists() or not load_dir_.is_dir():
            return None
        # Prefer files that contain the ticker symbol
        cand = sorted(list(load_dir_.glob(f"*{ticker_}*.pt")) + list(load_dir_.glob(f"*{ticker_}*.pth")))
        if cand:
            return cand[0]
        # Fallback: any .pt/.pth
        any_cand = sorted(list(load_dir_.glob("*.pt")) + list(load_dir_.glob("*.pth")))
        return any_cand[0] if any_cand else None

    @staticmethod
    def _list_all_tickers(tickers_path_: Path) -> List[str]:
        with open(tickers_path_, 'r') as f:
            tickers_json = json.load(f)
        return tickers_json["tickers"]

    @staticmethod
    def load_fnspid_news_(fnspid_news_: Union[str, Path], use_bodies_: bool) -> pl.LazyFrame:
        usecols = ["Date", "Article_title", "Stock_symbol"]
        rename = {"Article_title": "title", "Stock_symbol": "stock", "Date": "date"}
        dtypes = {
            "Date": pl.Utf8,
            "Article_title": pl.Utf8,
            "Stock_symbol": pl.Utf8,
        }
        additional_cols = []
        if use_bodies_:
            additional_cols = ["Article", "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]
            usecols.extend(additional_cols)
            for col in additional_cols:
                dtypes[col] = pl.Utf8

        # Use Polars lazy loading
        article_df = pl.scan_csv(
            fnspid_news_,
            has_header=True,
            schema_overrides=dtypes,
            try_parse_dates=False  # Parse manually later
        ).select(pl.col(usecols))
        Logger.info(f"Polars lazy dataframe schema:\n{article_df.schema=} {article_df.columns=}")
        if use_bodies_:
            # Concatenate body columns in lazy mode
            article_df = article_df.with_columns(
                pl.concat_str([pl.col(col) for col in additional_cols], separator=" ").alias("body")
            ).drop(additional_cols)

        # Rename columns
        article_df = article_df.rename(rename)
        return article_df

    @staticmethod
    def load_kaggle_news(kaggle_news_: Union[str, Path], use_bodies_: bool) -> pl.LazyFrame:
        usecols = ["title", "date", "stock"]
        dtypes = {"title": pl.Utf8, "date": pl.Utf8, "stock": pl.Utf8}
        article_df = pl.scan_csv(
            kaggle_news_,
            has_header=True,
            schema_overrides=dtypes,
            try_parse_dates=False
        ).select(pl.col(usecols))
        if use_bodies_:
            Logger.warning("No body column found in Kaggle CSV; falling back to titles only.")
            article_df = article_df.with_columns(pl.lit("").alias("body"))
        return article_df

    @staticmethod
    def load_and_filter_news(fnspid_news_: Union[str, Path], kaggle_news_: Union[str, Path], data_source_: str,
                             start_date_: pd.Timestamp, end_date_: pd.Timestamp, use_bodies_: bool) -> pl.LazyFrame:
        if data_source_ == "fnspid":
            article_df = Pipeline.load_fnspid_news_(fnspid_news_, use_bodies_)
            Logger.info(f"Loaded FNSPID news lazy dataframe with columns={article_df.columns} {article_df.width=}.")
            # Parse date to datetime with UTC time zone
            article_df = article_df.with_columns(
                pl.col("date").str.to_datetime(format="%Y-%m-%d %H:%M:%S %Z", strict=False, time_zone="UTC").alias("date")
            )
            Logger.info(f"Converted FNSPID 'date' column to datetime with UTC timezone. {pl.col('date').dt=}")
            article_df = article_df.filter(
                (pl.col("date") > pl.lit(start_date_)) & (pl.col("date") < pl.lit(end_date_))
            ).drop_nulls(subset=["date"])
            Logger.info(f"Filtered FNSPID news to date range {start_date_} - {end_date_}, resulting in lazy frame.")
        elif data_source_ == "kaggle":
            article_df = Pipeline.load_kaggle_news(kaggle_news_, use_bodies_)
            article_df = article_df.with_columns(
                pl.col("date").str.to_datetime(strict=False, time_zone="UTC").alias("date")
            )
            article_df = article_df.filter(
                (pl.col("date") > pl.lit(start_date_)) & (pl.col("date") < pl.lit(end_date_))
            ).drop_nulls(subset=["date"])
            Logger.info(f"Filtered Kaggle news to date range {start_date_} - {end_date_}, resulting in lazy frame.")
        else:
            raise ValueError(f"Invalid data_source: {data_source_}. Choose 'fnspid' or 'kaggle'.")
        return article_df

    @staticmethod
    def argparse() -> Tuple[argparse.Namespace, str]:
        runtime = datetime.now().strftime("%Y%m%d-%H%M%S")
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "--fnspid-csv-path", default="~/Projects/finBert/FNSPID/Stock_news/nasdaq_external_data.csv", #All_external.csv",
            help="FNSPID News CSV path")
        ap.add_argument(
            "--kaggle-csv-path", default="~/Projects/finBert/kaggle/analyst_ratings_processed.csv",
            help="Kaggle News CSV path")
        ap.add_argument(
            "--prices-dir", default="~/Projects/finBert/yf",
            help="Directory containing historical price CSVs named <TICKER>.csv")
        ap.add_argument(
            "--tickers-path", default="~/Projects/finBert/yf/tickers.json",
            help="Path to historical tickers.json"
        )
        ap.add_argument(
            "--sentiment-csv-path-in", required=False,
            help="If provided, load precomputed daily sentiment CSV from this path and skip FinBERT.")
        ap.add_argument(
            "--fine-tune-dir", default="finetuned_finbert",
            help="Fine-tune FinBERT store path")
        ap.add_argument(
            "--fine-tune-load-path", default=None,
            help="Fine-tune FinBERT load path")
        ap.add_argument(
            "--train", action="store_true", help="Train the predictor model"
        )
        ap.add_argument(
            "--data-source", default="kaggle", choices=["fnspid", "kaggle"],
            help="Data source: 'fnspid', 'kaggle'")
        ap.add_argument(
            "--ticker",
            required=True,
            type=str,
            nargs="+",
            help="Ticker(s) to train on (single stock or multiple tickers as a space-separated list, e.g., 'AAPL' or 'AAPL MSFT GOOGL')"
        )
        ap.add_argument(
            "--sentiment-threshold", type=float, default=0.8,
            help="Threshold for absolute sentiment score to consider an article as having sentiment")
        ap.add_argument(
            "--market-tz", default="America/New_York")
        ap.add_argument(
            "--market-close", default="16:30")
        ap.add_argument(
            "--batch-size", type=int, default=1, help="FinBERT scoring batch size")
        ap.add_argument(
            "--max-length", type=int, default=512, help="FinBERT tokenizer max_length")
        ap.add_argument(
            "--lookback", type=int, default=30)
        ap.add_argument(
            "--epochs", type=int, default=100)
        ap.add_argument(
            "--patience", type=int, default=30)
        ap.add_argument(
            "--use-bodies", action="store_true",
            help="Include article bodies in sentiment analysis if available")
        ap.add_argument(
            "--load-dir", default=None,
            help="Path to a weights file (.pt/.pth) or a directory containing saved weights to initialize training from.")
        ap.add_argument(
            "--lr", type=float, default=1e-5, help="Learning rate for Adam optimizer"
        )
        ap.add_argument(
            "--dropout-rate", type=float, default=0.00, help="Dropout rate for LSTM layers")
        ap.add_argument(
            "--weight-decay", type=float, default=0.00, help="Weight decay for Adam optimizer")
        ap.add_argument(
            "--predict-returns", action="store_true", default=False, help="Predict returns instead of closing prices")
        ap.add_argument(
            "--scale-method", default="minmax", choices=["minmax", "standard"],
            help="Use minmax or standard scaling for features")
        ap.add_argument(
            "--model", default="finbert-lstm", choices=["lstm", "rnn", "gru", "transformer", "tabmlp", "finbert-lstm", "finbert-rnn", "finbert-gru", "finbert-transformer", "finbert-tabmlp"],
            help="Model to use: 'lstm', 'rnn', 'gru', 'transformer', 'tabmlp', or their finbert variants")
        ap.add_argument(
            "--seed", type=int, default=1254, #1254
            help="Preset seed for deterministic training (sets Python, NumPy, and PyTorch seeds)")
        return ap.parse_args(), runtime

# standard vs minmax
# model: arima vs lstm vs transformer
# returns vs price
# compare model performance with sentiment vs without sentiment
# plots for data with sentiment vs all data
def main():
    Pipeline()

if __name__ == "__main__":
    main()
