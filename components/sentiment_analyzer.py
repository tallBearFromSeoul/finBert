import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from matplotlib.patches import Rectangle
from pathlib import Path

def visualize_csv(csv_path: Path, save_dir: Path, prices_dir: Path, yf_download_missing_data: bool = True):
    df = pd.read_csv(str(csv_path))
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    df = df.sort_values(['ticker', 'trading_date'])

    # For each ticker
    for ticker, group in df[df['ticker'].str[0].str.upper() >= 'S'].groupby('ticker'):
        # Load prices data
        prices_path = prices_dir.expanduser() / f"{ticker}.csv"
        if not prices_path.exists():
            print(f"Prices file not found for {ticker}: {prices_path}. Skipping price visualizations.")
            if yf_download_missing_data:
                print(f"Attempting to download data for {ticker} from Yahoo Finance.")
                ticker_data = yf.Ticker(ticker)
                hist = ticker_data.history(start='2005-10-14', end='2023-12-29', auto_adjust=False)
                if hist.empty:
                    print(f"No data found on Yahoo Finance for {ticker}. Skipping price visualizations.")
                    continue
                hist.reset_index(inplace=True)
                hist.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj close'}, inplace=True)
                if hist['date'].dtype.tz is not None:
                    hist['date'] = hist['date'].dt.tz_convert(None)
                    print(f"Converted timezone-aware dates to naive for {ticker}.")
                hist.to_csv(str(prices_path), index=False)
                print(f"Saved downloaded data to {prices_path}")
        prices = pd.read_csv(str(prices_path), parse_dates=['date'])
        prices['trading_date'] = prices['date']
        # Merge with sentiment group
        group = pd.merge(group, prices[['trading_date', 'open', 'high', 'low', 'close', 'adj close', 'volume']],
                         on='trading_date', how='left')
        # Compute daily returns
        group['returns'] = (group['adj close'] - group['adj close'].shift(1)) / group['adj close'].shift(1)
        # Create a single figure with GridSpec for subplots
        fig = plt.figure(figsize=(22, 18))
        gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2, 2, 2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        # Time series plot for SentimentScore with N_t as point sizes
        scatter = ax1.scatter(group['trading_date'], group['SentimentScore'], marker='o',
                              s=group['N_t'] * 50, alpha=0.7, color='blue', label='Sentiment Score')
        # Add vertical lines from each point to the x-axis
        for date, score, nt in zip(group['trading_date'], group['SentimentScore'], group['N_t']):
            ax1.vlines(x=date, ymin=min(0, score), ymax=max(0, score), colors='black', linestyles='solid',
                       linewidth=0.5, alpha=0.5)
            # Label points with N_t values, adjust vertical alignment for visibility
            va = 'bottom' if score >= 0 else 'top'
            ax1.text(date, score, f'{int(nt)}', fontsize=8, ha='right', va=va, alpha=0.8)
        ax1.set_title(f'Sentiment Score and Daily Returns over Time for {ticker} (Point Size ~ N_t)')
        ax1.set_xlabel('Trading Date')
        ax1.set_ylabel('Sentiment Score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.tick_params(axis='x', rotation=45) # Rotate x-ticks for visibility
        # Add grid and horizontal line at y=0
        ax1.grid(True)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        # Overlay daily returns on twin axis
        ax_returns = ax1.twinx()
        ax_returns.plot(group['trading_date'], group['returns'], color='red', label='Daily Returns', linewidth=1.5, alpha=0.8)
        ax_returns.set_ylabel('Daily Returns', color='red')
        ax_returns.tick_params(axis='y', labelcolor='red')
        # Create a combined legend for N_t sizes, Sentiment, and Returns
        nt_values = [min(group['N_t']), max(group['N_t'])] # Representative min and max N_t
        handles = [scatter] # Sentiment Score handle
        handles += [plt.scatter([], [], s=nt * 50, color='blue', alpha=0.7, marker='o')
                    for nt in nt_values]
        handles += [ax_returns.get_lines()[0]] # Returns handle
        labels = ['Sentiment Score'] + [f'N_t = {int(nt)}' for nt in nt_values] + ['Daily Returns']
        ax1.legend(handles, labels, title='Legend', loc='best')
        # Distribution histogram for SentimentScore
        # Calculate stats for title and filename
        n_points = len(group['SentimentScore'])
        min_score = group['SentimentScore'].min()
        max_score = group['SentimentScore'].max()
        mean_score = group['SentimentScore'].mean()
        # Single histogram call with centered bins and specified range
        ax2.hist(group['SentimentScore'], bins=100, range=(-1, 1), color='green', alpha=0.7)
        ax2.set_title(f'Distribution of Sentiment Score for {ticker} (N={n_points}, Min={min_score:.2f}, Max={max_score:.2f}, Mean={mean_score:.2f})')
        ax2.set_xlabel('Sentiment Score')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(-1, 1) # Set x-axis limits from -1 to 1
        ax2.grid(True) # Add grid for visibility
        # OHLC Candlestick and Volume plots
        plt.setp(ax3.get_xticklabels(), visible=False)
        col_up = 'green'
        col_down = 'red'
        up = group[group['close'] >= group['open']]
        down = group[group['close'] < group['open']]
        # Wicks and bodies for up candles
        if not up.empty:
            ax3.vlines(up['trading_date'], up['low'], up['high'], color=col_up, linewidth=1)
            ax3.bar(up['trading_date'], up['close'] - up['open'], width=8, bottom=up['open'], color=col_up, align='center')
        # Wicks and bodies for down candles
        if not down.empty:
            ax3.vlines(down['trading_date'], down['low'], down['high'], color=col_down, linewidth=1)
            ax3.bar(down['trading_date'], down['open'] - down['close'], width=8, bottom=down['close'], color=col_down, align='center')
        ax3.set_title(f'OHLC Candlestick Chart for {ticker}')
        ax3.set_ylabel('Price')
        ax3.grid(True)
        # Add lines for open, high, low, close in different colors
        line_open, = ax3.plot(group['trading_date'], group['open'], color='cyan', label='Open', linewidth=1)
        line_high, = ax3.plot(group['trading_date'], group['high'], color='lime', label='High', linewidth=1)
        line_low, = ax3.plot(group['trading_date'], group['low'], color='magenta', label='Low', linewidth=1)
        line_close, = ax3.plot(group['trading_date'], group['close'], color='blue', label='Close', linewidth=1)
        # Legend for OHLC including candles and lines
        handles = [
            Rectangle((0, 0), 4, 2, color=col_up, label='Bullish'),
            Rectangle((0, 0), 4, 2, color=col_down, label='Bearish'),
            line_open,
            line_high,
            line_low,
            line_close
        ]
        ax3.legend(handles=handles, loc='best')
        # Volume
        if not up.empty:
            ax4.bar(up['trading_date'], up['volume'], width=4, color=col_up, align='center', label='Bullish Volume')
        if not down.empty:
            ax4.bar(down['trading_date'], down['volume'], width=4, color=col_down, align='center', label='Bearish Volume')
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Trading Date')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True)
        ax4.legend(loc='best')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Save with stats in filename
        plt.savefig(save_dir / f"{ticker}_sentiment_analysis_N{n_points}_min{min_score:.2f}_max{max_score:.2f}_mean{mean_score:.2f}.png")
        plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize sentiment CSV data.")
    parser.add_argument("--csv-path", type=Path, help="Path to the sentiment CSV file.")
    parser.add_argument("--save-dir", type=Path, help="Directory to save the visualizations.")
    parser.add_argument("--prices-dir", type=Path, default="~/Projects/finBert/FNSPID/Stock_price/full_history",
                        help="Directory with price CSV files per ticker.")

    args = parser.parse_args()

    visualize_csv(args.csv_path, args.save_dir, args.prices_dir)
