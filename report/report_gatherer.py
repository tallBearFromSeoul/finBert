import json
import matplotlib.pyplot as plt
import numpy as np

def plot_comparisons(stock_data, stock, output_file=None):
    """
    Generate a beautiful plot comparing metrics (MSE, MAE, RMSE) for Vanilla vs Sentiment across models for a given stock, including train, val, and test.
    """
    splits = ['Train', 'Val', 'Test']
    metrics = ['MSE', 'MAE', 'RMSE']
    models = ['RNN', 'LSTM', 'TabMLP']
    variants = ['Vanilla', 'Sentiment']

    fig, axs = plt.subplots(len(splits), len(metrics), figsize=(15, 12), sharey=False)
    fig.suptitle(f'Comparison of Metrics for {stock}', fontsize=16)

    bar_width = 0.35
    x = np.arange(len(models))

    colors = {'Vanilla': 'skyblue', 'Sentiment': 'lightgreen'}

    for i, split in enumerate(splits):
        for j, metric in enumerate(metrics):
            ax = axs[i, j]
            vanilla_values = [stock_data[model]['Vanilla'][f'{split} {metric}'] for model in models]
            sentiment_values = [stock_data[model]['Sentiment'][f'{split} {metric}'] for model in models]

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
            ax.set_yscale('log')  # Log scale since values are small
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

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__ == "__main__":

    data_sources = {
        'NFLX': {
            'RNN': {
                'Vanilla': 'report/rnn/20250825-182559/',
                'Sentiment': 'report/rnn/20250825-182653/'
            },
            'LSTM': {
                'Vanilla': 'report/lstm/20250824-234116/',
                'Sentiment': 'report/lstm/20250824-234258/'
            },
            'TabMLP': {
                'Vanilla': 'report/tabmlp/20250825-183844/',
                'Sentiment': 'report/tabmlp/20250825-183926/'
            }
        },
        'HD': {
            'RNN': {
                'Vanilla': 'report/rnn/20250825-182905/',
                'Sentiment': 'report/rnn/20250825-182756/'
            },
            'LSTM': {
                'Vanilla': 'report/lstm/20250824-234926/',
                'Sentiment': 'report/lstm/20250824-234754/'
            },
            'TabMLP': {
                'Vanilla': 'report/tabmlp/20250825-183725/',
                'Sentiment': 'report/tabmlp/20250825-183627/'
            }
        }
    }

    stocks = ['NFLX', 'HD']
    models = ['RNN', 'LSTM', 'TabMLP']
    variants = ['Vanilla', 'Sentiment']
    metrics = ['MSE', 'MAE', 'RMSE']

    # Collect data
    all_data = {}
    for stock in stocks:
        all_data[stock] = {}
        print(f"#### {stock}:")
        for model in models:
            all_data[stock][model] = {}
            for variant in variants:
                dir_path = data_sources[stock][model][variant]
                json_path = dir_path + 'evaluation.json'
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Assuming the JSON is a list with one dict
                    d = data[0]
                    # Verify ticker
                    if d['ticker'] != stock:
                        print(f"Warning: Ticker mismatch in {json_path}: expected {stock}, got {d['ticker']}")

                all_data[stock][model][variant] = {
                    'Train MSE': d['best_train_mse'],
                    'Val MSE': d['best_val_mse'],
                    'Test MSE': d['test_MSE'],
                    'Train MAE': d['best_train_mae'],
                    'Val MAE': d['best_val_mae'],
                    'Test MAE': d['test_MAE'],
                    'Train RMSE': d['best_train_rmse'],
                    'Val RMSE': d['best_val_rmse'],
                    'Test RMSE': d['test_RMSE']
                }

                # Format for printing
                train_mse_fmt = '{:.4e}'.format(d['best_train_mse'])
                val_mse_fmt = '{:.4e}'.format(d['best_val_mse'])
                test_mse_fmt = '{:.4e}'.format(d['test_MSE'])
                train_mae_fmt = '{:.4e}'.format(d['best_train_mae'])
                val_mae_fmt = '{:.4e}'.format(d['best_val_mae'])
                test_mae_fmt = '{:.4e}'.format(d['test_MAE'])
                train_rmse_fmt = '{:.4e}'.format(d['best_train_rmse'])
                val_rmse_fmt = '{:.4e}'.format(d['best_val_rmse'])
                test_rmse_fmt = '{:.4e}'.format(d['test_RMSE'])

                print(f"| {model} | {variant} | {train_mse_fmt} | {val_mse_fmt} | {test_mse_fmt} | {train_mae_fmt} | {val_mae_fmt} | {test_mae_fmt} | {train_rmse_fmt} | {val_rmse_fmt} | {test_rmse_fmt} |")
        print("\n")  # Separator between stocks

    # Generate plots for each stock
    for stock in stocks:
        plot_comparisons(all_data[stock], stock, output_file=f'report/{stock}_metrics_comparison.png')
