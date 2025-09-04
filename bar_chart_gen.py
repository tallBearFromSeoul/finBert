import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    # Read the table from stdin
    table_str = (
"""
| RNN | Vanilla | 1.4127e-05 | 2.1847e-06 | 4.0529e-06 |
| RNN | Sentiment | 9.6809e-06 | 1.8499e-06 | 1.9911e-06 |
| GRU | Vanilla | 1.2065e-05 | 3.2257e-06 | 4.4572e-06 |
| GRU | Sentiment | 1.3904e-05 | 3.2427e-06 | 4.2571e-06 |
| Transformer | Vanilla | 3.3365e-05 | 3.5828e-06 | 9.8491e-06 |
| Transformer | Sentiment | 2.1850e-05 | 2.6842e-06 | 4.9500e-06 |
| LSTM | Vanilla | 4.4676e-05 | 9.1531e-06 | 1.6366e-05 |
| LSTM | Sentiment | 2.3496e-05 | 5.6960e-06 | 9.3376e-06 |
| TabMLP | Vanilla | 4.2090e-05 | 8.9945e-06 | 1.1393e-05 |
| TabMLP | Sentiment | 9.2689e-06 | 3.7506e-06 | 2.0314e-06 |
""")

    # Parse the table
    lines = table_str.strip().split('\n')
    data = []
    for line in lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if parts:
            model, variant, train, val, test = parts
            data.append((model, variant, float(train), float(val), float(test)))
    # Extract models and MSE lists assuming order: Vanilla then Sentiment for each model
    models = [row[0] for row in data[::2]]
    train_mse_vanilla = [row[2] for row in data[::2]]
    train_mse_sentiment = [row[2] for row in data[1::2]]
    val_mse_vanilla = [row[3] for row in data[::2]]
    val_mse_sentiment = [row[3] for row in data[1::2]]
    test_mse_vanilla = [row[4] for row in data[::2]]
    test_mse_sentiment = [row[4] for row in data[1::2]]
    # Set up the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(models))
    # Calculate max values for ylims
    max_train = max(max(train_mse_vanilla), max(train_mse_sentiment))
    max_val = max(max(val_mse_vanilla), max(val_mse_sentiment))
    max_test = max(max(test_mse_vanilla), max(test_mse_sentiment))
    # Train MSE subplot
    rects1 = ax1.bar(x - bar_width/2, train_mse_vanilla, bar_width, label='Vanilla', color='#4e79a7')
    rects2 = ax1.bar(x + bar_width/2, train_mse_sentiment, bar_width, label='Sentiment', color='#f28e2b')
    ax1.set_ylabel('Train MSE')
    ax1.set_title('Train MSE Comparison: Vanilla vs Sentiment Models')
    ax1.set_ylim(0, max_train * 1.5)
    ax1.legend()
    ax1.grid(True)
    labels_v = [f"{models[i]} Vanilla\n{train_mse_vanilla[i]:.2e}" for i in range(len(models))]
    labels_s = [f"{models[i]} FinBERT\n{train_mse_sentiment[i]:.2e}" for i in range(len(models))]
    ax1.bar_label(rects1, labels=labels_v, padding=3, rotation=90)
    ax1.bar_label(rects2, labels=labels_s, padding=3, rotation=90)
    # Val MSE subplot
    rects1 = ax2.bar(x - bar_width/2, val_mse_vanilla, bar_width, label='Vanilla', color='#4e79a7')
    rects2 = ax2.bar(x + bar_width/2, val_mse_sentiment, bar_width, label='Sentiment', color='#f28e2b')
    ax2.set_ylabel('Val MSE')
    ax2.set_title('Val MSE Comparison: Vanilla vs Sentiment Models')
    ax2.set_ylim(0, max_val * 1.5)
    ax2.legend()
    ax2.grid(True)
    labels_v = [f"{models[i]} Vanilla\n{val_mse_vanilla[i]:.2e}" for i in range(len(models))]
    labels_s = [f"{models[i]} FinBERT\n{val_mse_sentiment[i]:.2e}" for i in range(len(models))]
    ax2.bar_label(rects1, labels=labels_v, padding=3, rotation=90)
    ax2.bar_label(rects2, labels=labels_s, padding=3, rotation=90)
    # Test MSE subplot
    rects1 = ax3.bar(x - bar_width/2, test_mse_vanilla, bar_width, label='Vanilla', color='#4e79a7')
    rects2 = ax3.bar(x + bar_width/2, test_mse_sentiment, bar_width, label='Sentiment', color='#f28e2b')
    ax3.set_ylabel('Test MSE')
    ax3.set_title('Test MSE Comparison: Vanilla vs Sentiment Models')
    ax3.set_ylim(0, max_test * 1.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_xlabel('Model')
    ax3.legend()
    ax3.grid(True)
    labels_v = [f"{models[i]} Vanilla\n{test_mse_vanilla[i]:.2e}" for i in range(len(models))]
    labels_s = [f"{models[i]} FinBERT\n{test_mse_sentiment[i]:.2e}" for i in range(len(models))]
    ax3.bar_label(rects1, labels=labels_v, padding=3, rotation=90)
    ax3.bar_label(rects2, labels=labels_s, padding=3, rotation=90)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.savefig("mse_bar_chart.png")
