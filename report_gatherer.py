from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess

from utils.logger import Logger
from utils.pathlib_utils import ensure_dir

def parse_output_dir(stdout_):
    lines = stdout_.splitlines()
    for line in lines:
        if line.startswith("[Logger] writing to PosixPath('"):
            # Extract the path inside the quotes
            path_start = line.find("'") + 1
            path_end = line.find("'", path_start)
            log_path = line[path_start:path_end]
            # The output dir is the parent of logs
            output_dir = os.path.dirname(os.path.dirname(log_path))
            return output_dir
    raise ValueError("Could not find logger output path in stdout")

def generate_reports():
    all_tickers = True
    tickers = ["BMY", "EBAY", "EWI", "BABA", "DAL", "JNJ", "NFLX", "TSLA"]
    models = ["gru", "finbert-gru", "transformer", "finbert-transformer",
              "rnn", "finbert-rnn", "lstm", "finbert-lstm", "tabmlp", "finbert-tabmlp"]
    model_map = {"gru": "GRU", "transformer": "Transformer", "finbert-transformer": "Transformer",
                 "rnn": "RNN", "lstm": "LSTM", "tabmlp": "TabMLP"}
    data_sources = {"RNN": {}, "LSTM": {}, "TabMLP": {}, "Transformer": {}, "GRU": {}}
    sentiment_csv_path = "output/20250823-140342/sentiment_daily.csv"
    for model in models:
        base_model = model.replace("finbert-", "") if "finbert-" in model else model
        variant = "Sentiment" if "finbert-" in model else "Vanilla"
        if all_tickers:
            cmd = [
                "python3", "-m", "components.pipeline",
                "--ticker", "all-tickers",
                "--scale-method", "minmax",
                "--data-source", "kaggle",
                "--model", model,
                "--sentiment-csv-path", sentiment_csv_path,
                #--predict-returns
            ]
        else:
            cmd = [
                "python3", "-m", "components.pipeline",
                "--ticker"
            ] + [str(ticker) for ticker in tickers] + [  # Pass tickers as separate arguments
                "--scale-method", "minmax",
                "--data-source", "kaggle",
                "--model", model,
                "--sentiment-csv-path", sentiment_csv_path,
                #--predict-returns
            ]
        Logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            Logger.info(f"Error running command for {cmd} {model}: {result.stderr}")
            continue
        output_dir = parse_output_dir(result.stdout)
        timestamp = os.path.basename(output_dir)
        report_dir = f'report/{base_model}/{timestamp}/'
        os.makedirs(f'report/{base_model}', exist_ok=True)
        shutil.copytree(output_dir, report_dir, dirs_exist_ok=True)
        Logger.info(f"Copied output to {report_dir}")
        upper_model = model_map[base_model]
        data_sources[upper_model][variant] = report_dir
    # Save the json for reference, though we use in memory
    with open('report/report_paths.json', 'w') as f:
        json.dump(data_sources, f, indent=4)
    return data_sources

def plot_comparisons(stock_data_, stocks_, output_file_):
    """
    Generate a beautiful plot comparing metrics (MSE, MAE, RMSE) for Vanilla vs Sentiment across models for a given stock, including train, val, and test.
    """
    splits = ['Train', 'Val', 'Test']
    metric = 'MSE'
    models = ['RNN', 'GRU', 'LSTM', 'TabMLP', 'Transformer']
    variants = ['Vanilla', 'Sentiment']
    fig, axs = plt.subplots(len(splits), 1, figsize=(15, 12), sharey=False)
    fig.suptitle(f'Comparison of Metrics for \n{stocks_}', fontsize=12)
    bar_width = 0.35
    x = np.arange(len(models))
    colors = {'Vanilla': 'skyblue', 'Sentiment': 'lightgreen'}
    for i, split in enumerate(splits):
        ax = axs[i]
        vanilla_values = [stock_data_[model]['Vanilla'][f'{split} {metric}'] for model in models]
        sentiment_values = [stock_data_[model]['Sentiment'][f'{split} {metric}'] for model in models]
        ax.bar(x - bar_width/2, vanilla_values, bar_width, label='Vanilla', color=colors['Vanilla'])
        ax.bar(x + bar_width/2, sentiment_values, bar_width, label='Sentiment', color=colors['Sentiment'])
        if i == 0:
            ax.set_title(f'{metric}')
        if j == 0:
            ax.set_ylabel(split)
        ax.set_xticks(x)
        if i == len(splits) - 1:
            ax.set_xticklabels(models)
        else:
            ax.set_xticklabels([])
        ax.set_yscale('log') # Log scale since values are small
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Add value labels
        for k, v in enumerate(vanilla_values):
            ax.text(k - bar_width/2, v, '{:.2e}'.format(v), ha='center', va='bottom', fontsize=8)
        for k, v in enumerate(sentiment_values):
            ax.text(k + bar_width/2, v, '{:.2e}'.format(v), ha='center', va='bottom', fontsize=8)
    # Add legend to the figure
    handles = [plt.Rectangle((0,0),1,1, color=colors[v]) for v in variants]
    fig.legend(handles, variants, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_file_:
        plt.savefig(output_file_)
    else:
        plt.show()

if __name__ == "__main__":
    ensure_dir(Path("report"))
    ensure_dir(Path("report/rnn"))
    ensure_dir(Path("report/lstm"))
    ensure_dir(Path("report/tabmlp"))
    data_sources = generate_reports()
    Logger.info("Data sources for reports: {data_sources}")
    stocks = list(data_sources.keys())
    models = ['RNN', 'LSTM', 'TabMLP']
    variants = ['Vanilla', 'Sentiment']
    # Collect data
    all_data = {}
    for model in models:
        all_data[model] = {}
        for variant in variants:
            dir_path = data_sources[model][variant]
            json_path = dir_path + 'evaluation.json'
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Assuming the JSON is a list with one dict
                d = data[0]
            all_data[model][variant] = {
                'Train MSE': d['best_train_mse_scaled'],
                'Val MSE': d['best_val_mse_scaled'],
                'Test MSE': d['test_MSE_scaled'],
            }
            # Format for Logger.infoing
            train_mse_fmt = '{:.4e}'.format(d['best_train_mse_scaled'])
            val_mse_fmt = '{:.4e}'.format(d['best_val_mse_scaled'])
            test_mse_fmt = '{:.4e}'.format(d['test_MSE_scaled'])
            Logger.info(f"| {model} | {variant} | {train_mse_fmt} | {val_mse_fmt} | {test_mse_fmt} |")
    Logger.info("\n") # Separator between stocks
    # Generate plots for each stock
    plot_comparisons(all_data, stocks, f'report/metrics_comparison.png')
