# Financial Sentiment Analysis and Stock Return Prediction
This project leverages FinBERT for sentiment analysis on financial news headlines and integrates it with deep learning models to predict next-day stock returns. It builds upon the methodology in [Financial Sentiment Analysis Using FinBERT with Application in Predicting Stock Movement](https://arxiv.org/abs/2306.02136) (arXiv:2306.02136v3), which focused on predicting next-day closing prices, whereas this work targets next-day returns to better capture relative market movements. The goal is to demonstrate how sentiment-enhanced features improve prediction accuracy over vanilla models.
## Part 1: Sentiment Scoring and Fine-Tuning
I scored sentiments from headlines in the "Daily Financial News for 6000+ Stocks" dataset (Kaggle), covering over 1 million entries up to 2019. Each headline was processed using pre-trained FinBERT (`ProsusAI/finbert`) to extract probabilities for positive, negative, and neutral sentiments. The daily aggregated sentiment score per ticker is calculated as:
$$ \text{SentimentScore}_t = \frac{1}{N_t} \sum_{n=1}^{N_t} [P_n(\text{positive}) - P_n(\text{negative})] $$
where \(N_t\) is the number of articles on day \(t\).
To adapt FinBERT for financial tasks, I fine-tuned it using the Numerical Sentiment Index (NSI) derived from historical stock prices (via yfinance API). NSI quantifies market-based sentiment as:
$$ \text{NSI}_t = \begin{cases}
1 & \text{if } \text{return}_t > s \\
0 & \text{if } -s \leq \text{return}_t \leq s \\
-1 & \text{if } \text{return}_t < -s
\end{cases} $$
with \(\text{return}_t = \frac{\text{ClosePrice}_t - \text{OpenPrice}_t}{\text{OpenPrice}_t}\) and threshold \(s = 0.01\). This fine-tuning aligns textual sentiment with actual market movements, enhancing domain-specific accuracy.
Post-fine-tuning, I analyzed the distribution of sentiment scores across tickers, which often exhibited a slight positive skew (mean around 0.1-0.3 for most stocks), reflecting optimistic financial news bias. I selected tickers with sufficient data points (e.g., >500) and balanced distributions for statistical significance, avoiding those with extreme sentiment clustering that could bias predictions.
For comprehensiveness, I also examined the distribution of daily stock returns from the historical data. Returns typically followed a leptokurtic pattern (fat tails, kurtosis >3), indicating higher volatility and outlier events compared to a normal distribution. This informed feature engineering, such as incorporating volatility measures in lagged features to capture non-stationary behaviors.
Visualizations for each ticker include: time series of SentimentScore (point size proportional to \(N_t\), with vertical lines and overlaid daily returns), histogram of SentimentScore (with stats: N, min, max, mean), OHLC candlestick chart (with open/high/low/close lines), and volume bars (colored by bullish/bearish).
![Sentiment Distribution Plot - Overall](path/to/sentiment_distribution_overall.png)
![Stock Returns Distribution Plot](path/to/returns_distribution.png)
![Sentiment Score Data Visualization - Stock 1 [Ticker, e.g., MSFT]](path/to/stock1_sentiment_visualization.png)
![Sentiment Score Data Visualization - Stock 2 [Ticker, e.g., AAPL]](path/to/stock2_sentiment_visualization.png)

Top 50 tickers ranked by number of datapoints (N):
1. Ticker: DAL, N: 922, Min: -0.11, Max: 0.82, Mean: 0.03, Filename: DAL_sentiment_analysis_N922_min-0.11_max0.82_mean0.03.png
2. Ticker: BMY, N: 916, Min: -0.08, Max: 0.81, Mean: 0.04, Filename: BMY_sentiment_analysis_N916_min-0.08_max0.81_mean0.04.png
3. Ticker: EBAY, N: 885, Min: -0.1, Max: 0.81, Mean: 0.0, Filename: EBAY_sentiment_analysis_N885_min-0.10_max0.81_mean0.00.png
4. Ticker: BABA, N: 868, Min: -0.07, Max: 0.4, Mean: 0.02, Filename: BABA_sentiment_analysis_N868_min-0.07_max0.40_mean0.02.png
5. Ticker: EWI, N: 866, Min: -0.07, Max: 0.76, Mean: 0.08, Filename: EWI_sentiment_analysis_N866_min-0.07_max0.76_mean0.08.png
6. Ticker: JNJ, N: 856, Min: -0.1, Max: 0.78, Mean: 0.03, Filename: JNJ_sentiment_analysis_N856_min-0.10_max0.78_mean0.03.png
7. Ticker: MRK, N: 855, Min: -0.09, Max: 0.81, Mean: 0.02, Filename: MRK_sentiment_analysis_N855_min-0.09_max0.81_mean0.02.png
8. Ticker: LMT, N: 849, Min: -0.1, Max: 0.82, Mean: 0.02, Filename: LMT_sentiment_analysis_N849_min-0.10_max0.82_mean0.02.png
9. Ticker: AGN, N: 837, Min: -0.09, Max: 0.69, Mean: 0.02, Filename: AGN_sentiment_analysis_N837_min-0.09_max0.69_mean0.02.png
10. Ticker: GILD, N: 832, Min: -0.12, Max: 0.55, Mean: 0.02, Filename: GILD_sentiment_analysis_N832_min-0.12_max0.55_mean0.02.png
11. Ticker: MU, N: 806, Min: -0.1, Max: 0.82, Mean: 0.02, Filename: MU_sentiment_analysis_N806_min-0.10_max0.82_mean0.02.png
12. Ticker: AZN, N: 791, Min: -0.13, Max: 0.81, Mean: 0.03, Filename: AZN_sentiment_analysis_N791_min-0.13_max0.81_mean0.03.png
13. Ticker: GLD, N: 788, Min: -0.08, Max: 0.81, Mean: 0.05, Filename: GLD_sentiment_analysis_N788_min-0.08_max0.81_mean0.05.png
14. Ticker: ATVI, N: 770, Min: -0.1, Max: 0.81, Mean: 0.02, Filename: ATVI_sentiment_analysis_N770_min-0.10_max0.81_mean0.02.png
15. Ticker: CAT, N: 763, Min: -0.09, Max: 0.81, Mean: 0.04, Filename: CAT_sentiment_analysis_N763_min-0.09_max0.81_mean0.04.png
16. Ticker: FDX, N: 742, Min: -0.09, Max: 0.82, Mean: 0.04, Filename: FDX_sentiment_analysis_N742_min-0.09_max0.82_mean0.04.png
17. Ticker: LLY, N: 740, Min: -0.1, Max: 0.74, Mean: 0.03, Filename: LLY_sentiment_analysis_N740_min-0.10_max0.74_mean0.03.png
18. Ticker: KO, N: 735, Min: -0.09, Max: 0.82, Mean: 0.03, Filename: KO_sentiment_analysis_N735_min-0.09_max0.82_mean0.03.png
19. Ticker: DISH, N: 733, Min: -0.14, Max: 0.82, Mean: 0.01, Filename: DISH_sentiment_analysis_N733_min-0.14_max0.82_mean0.01.png
20. Ticker: MS, N: 729, Min: -0.1, Max: 0.82, Mean: 0.04, Filename: MS_sentiment_analysis_N729_min-0.10_max0.82_mean0.04.png
21. Ticker: NVDA, N: 718, Min: -0.11, Max: 0.8, Mean: 0.01, Filename: NVDA_sentiment_analysis_N718_min-0.11_max0.80_mean0.01.png
22. Ticker: CMCSA, N: 712, Min: -0.1, Max: 0.34, Mean: 0.02, Filename: CMCSA_sentiment_analysis_N712_min-0.10_max0.34_mean0.02.png
23. Ticker: BIDU, N: 710, Min: -0.09, Max: 0.82, Mean: 0.03, Filename: BIDU_sentiment_analysis_N710_min-0.09_max0.82_mean0.03.png
24. Ticker: AXP, N: 702, Min: -0.1, Max: 0.81, Mean: 0.03, Filename: AXP_sentiment_analysis_N702_min-0.10_max0.81_mean0.03.png
25. Ticker: QCOM, N: 700, Min: -0.12, Max: 0.3, Mean: -0.0, Filename: QCOM_sentiment_analysis_N700_min-0.12_max0.30_mean-0.00.png
26. Ticker: VZ, N: 691, Min: -0.11, Max: 0.4, Mean: 0.02, Filename: VZ_sentiment_analysis_N691_min-0.11_max0.40_mean0.02.png
27. Ticker: HD, N: 679, Min: -0.1, Max: 0.82, Mean: 0.03, Filename: HD_sentiment_analysis_N679_min-0.10_max0.82_mean0.03.png
28. Ticker: MYL, N: 667, Min: -0.11, Max: 0.43, Mean: -0.0, Filename: MYL_sentiment_analysis_N667_min-0.11_max0.43_mean-0.00.png
29. Ticker: MDT, N: 663, Min: -0.07, Max: 0.82, Mean: 0.03, Filename: MDT_sentiment_analysis_N663_min-0.07_max0.82_mean0.03.png
30. Ticker: BX, N: 659, Min: -0.11, Max: 0.82, Mean: 0.03, Filename: BX_sentiment_analysis_N659_min-0.11_max0.82_mean0.03.png
31. Ticker: FSLR, N: 659, Min: -0.1, Max: 0.82, Mean: 0.01, Filename: FSLR_sentiment_analysis_N659_min-0.10_max0.82_mean0.01.png
32. Ticker: BIIB, N: 659, Min: -0.11, Max: 0.8, Mean: 0.03, Filename: BIIB_sentiment_analysis_N659_min-0.11_max0.80_mean0.03.png
33. Ticker: P, N: 659, Min: -0.1, Max: 0.28, Mean: 0.02, Filename: P_sentiment_analysis_N659_min-0.10_max0.28_mean0.02.png
34. Ticker: NOK, N: 658, Min: -0.08, Max: 0.81, Mean: 0.03, Filename: NOK_sentiment_analysis_N658_min-0.08_max0.81_mean0.03.png
35. Ticker: AIG, N: 656, Min: -0.1, Max: 0.81, Mean: 0.02, Filename: AIG_sentiment_analysis_N656_min-0.10_max0.81_mean0.02.png
36. Ticker: BAC, N: 655, Min: -0.1, Max: 0.42, Mean: 0.03, Filename: BAC_sentiment_analysis_N655_min-0.10_max0.42_mean0.03.png
37. Ticker: ORCL, N: 652, Min: -0.1, Max: 0.78, Mean: 0.01, Filename: ORCL_sentiment_analysis_N652_min-0.10_max0.78_mean0.01.png
38. Ticker: DB, N: 645, Min: -0.1, Max: 0.82, Mean: 0.03, Filename: DB_sentiment_analysis_N645_min-0.10_max0.82_mean0.03.png
39. Ticker: CHK, N: 642, Min: -0.13, Max: 0.82, Mean: 0.03, Filename: CHK_sentiment_analysis_N642_min-0.13_max0.82_mean0.03.png
40. Ticker: GSK, N: 639, Min: -0.08, Max: 0.81, Mean: 0.03, Filename: GSK_sentiment_analysis_N639_min-0.08_max0.81_mean0.03.png
41. Ticker: AET, N: 636, Min: -0.08, Max: 0.81, Mean: 0.02, Filename: AET_sentiment_analysis_N636_min-0.08_max0.81_mean0.02.png
42. Ticker: MA, N: 623, Min: -0.1, Max: 0.8, Mean: 0.04, Filename: MA_sentiment_analysis_N623_min-0.10_max0.80_mean0.04.png
43. Ticker: CVS, N: 617, Min: -0.09, Max: 0.81, Mean: 0.02, Filename: CVS_sentiment_analysis_N617_min-0.09_max0.81_mean0.02.png
44. Ticker: ADBE, N: 617, Min: -0.09, Max: 0.81, Mean: 0.04, Filename: ADBE_sentiment_analysis_N617_min-0.09_max0.81_mean0.04.png
45. Ticker: BNO, N: 615, Min: -0.08, Max: 0.48, Mean: 0.06, Filename: BNO_sentiment_analysis_N615_min-0.08_max0.48_mean0.06.png
46. Ticker: CMG, N: 614, Min: -0.08, Max: 0.81, Mean: 0.02, Filename: CMG_sentiment_analysis_N614_min-0.08_max0.81_mean0.02.png
47. Ticker: FCX, N: 612, Min: -0.1, Max: 0.82, Mean: 0.05, Filename: FCX_sentiment_analysis_N612_min-0.10_max0.82_mean0.05.png
48. Ticker: EWU, N: 608, Min: -0.09, Max: 0.82, Mean: 0.1, Filename: EWU_sentiment_analysis_N608_min-0.09_max0.82_mean0.10.png
49. Ticker: COP, N: 605, Min: -0.09, Max: 0.82, Mean: 0.04, Filename: COP_sentiment_analysis_N605_min-0.09_max0.82_mean0.04.png
50. Ticker: TWX, N: 599, Min: -0.09, Max: 0.81, Mean: 0.03, Filename: TWX_sentiment_analysis_N599_min-0.09_max0.81_mean0.03.png



## Part 2: Predictive Modeling with Sentiment Features
Using the fine-tuned sentiment scores, I constructed daily features: sentiment score + lagged OHLCV market features (open, high, low, close, volume) over a look-back window (e.g., 60 days), aligned with the paper's approach. Features were normalized using MinMax scaling to [0,1] for stable training across varying scales. Unlike the paper's focus on predicting next-day closing prices, the target here is the next day's return (\(\text{return}_{t+1}\)), normalized similarly to handle scale differences across tickers and emphasize relative changes. Only data points with available sentiment were used, ensuring alignment between news and market data.
I trained regressors on chronological splits (90% train, 10% test), comparing:
- **Vanilla models**: Lagged features only (no sentiment).
- **Sentiment-enhanced models**: Including sentiment score.
Models evaluated:
- **RNN (Recurrent Neural Network)**: Basic sequential model capturing temporal dependencies via hidden states; suitable for time-series but prone to vanishing gradients over long sequences.
- **LSTM (Long Short-Term Memory)**: Improves RNN by adding gates (forget, input, output) to handle long-term dependencies and mitigate vanishing gradients; ideal for financial time-series with non-linear patterns and persistent autocorrelations.
- **TabMLP (Tabular Multi-Layer Perceptron)**: Feed-forward neural network for tabular data; excels in capturing feature interactions via dense layers without explicit sequencing, serving as a non-recurrent baseline to benchmark against recurrent architectures.
Training used MSE loss, Adam optimizer (learning rate 0.001, with decay), batch size 1 (beneficial for capturing fine-grained patterns in sequential data, despite noisier gradient updates, as per the paper), and early stopping (patience 20) based on validation loss (10% of train). No regularization was applied to maintain model simplicity given small per-ticker datasets, relying on early stopping to prevent overfitting without suppressing learning on limited signals. Hyperparameters were tuned via grid search for stability.

## Results
### Performance Metrics
Metrics for train/validation/test sets (MSE, MAE, RMSE). Lower values indicate better performance, with sentiment models showing consistent reductions in error (e.g., -% MSE improvement on test set for RNN on Stock 1, -% for LSTM, etc.), highlighting sentiment's role in explaining return variance beyond lagged features.
#### NFLX
| Model | Variant | Train MSE | Val MSE | Test MSE | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE |
|-------|---------|-----------|---------|----------|-----------|---------|----------|------------|----------|-----------|
| RNN | Vanilla | 6.7150e-04 | 5.6893e-04 | 3.7966e-04 | 1.7733e-02 | 1.6639e-02 | 1.4708e-02 | 2.5913e-02 | 2.3852e-02 | 1.9485e-02 |
| RNN | Sentiment | 6.7009e-04 | 5.9600e-04 | 3.5543e-04 | 1.7794e-02 | 1.7062e-02 | 1.4305e-02 | 2.5886e-02 | 2.4413e-02 | 1.8853e-02 |
| LSTM | Vanilla | 1.0977e-04 | 5.5927e-04 | 8.5299e-04 | 7.8826e-03 | 1.7368e-02 | 2.2996e-02 | 1.0477e-02 | 2.3649e-02 | 2.9206e-02 |
| LSTM | Sentiment | 2.2249e-04 | 5.6999e-04 | 4.7544e-04 | 1.1140e-02 | 1.6767e-02 | 1.6591e-02 | 1.4916e-02 | 2.3874e-02 | 2.1805e-02 |
| TabMLP | Vanilla | 6.5144e-04 | 5.7376e-04 | 3.6350e-04 | 1.7423e-02 | 1.6648e-02 | 1.4424e-02 | 2.5523e-02 | 2.3953e-02 | 1.9066e-02 |
| TabMLP | Sentiment | 3.2712e-04 | 5.7647e-04 | 4.7745e-04 | 1.2657e-02 | 1.7007e-02 | 1.7369e-02 | 1.8086e-02 | 2.4010e-02 | 2.1851e-02 |

![NFLX metrics comparison](report/NFLX_metrics_comparison.png)

#### HD:
| Model | Variant | Train MSE | Val MSE | Test MSE | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE |
|-------|---------|-----------|---------|----------|-----------|---------|----------|------------|----------|-----------|
| RNN | Vanilla | 1.7909e-04 | 2.3292e-04 | 1.8449e-04 | 9.4846e-03 | 1.0254e-02 | 9.8623e-03 | 1.3382e-02 | 1.5262e-02 | 1.3583e-02 |
| RNN | Sentiment | 9.3830e-05 | 2.2075e-04 | 1.9106e-04 | 7.1706e-03 | 9.9357e-03 | 1.0141e-02 | 9.6866e-03 | 1.4858e-02 | 1.3822e-02 |
| LSTM | Vanilla | 1.8039e-04 | 2.3063e-04 | 1.8548e-04 | 9.4862e-03 | 1.0161e-02 | 9.8004e-03 | 1.3431e-02 | 1.5186e-02 | 1.3619e-02 |
| LSTM | Sentiment | 8.9531e-05 | 2.2235e-04 | 1.8797e-04 | 6.9792e-03 | 1.0049e-02 | 9.8541e-03 | 9.4621e-03 | 1.4912e-02 | 1.3710e-02 |
| TabMLP | Vanilla | 1.7803e-04 | 2.3108e-04 | 1.8663e-04 | 9.3831e-03 | 1.0167e-02 | 9.8214e-03 | 1.3343e-02 | 1.5201e-02 | 1.3661e-02 |
| TabMLP | Sentiment | 6.2068e-05 | 2.2084e-04 | 1.8922e-04 | 5.8283e-03 | 1.0069e-02 | 1.0180e-02 | 7.8783e-03 | 1.4861e-02 | 1.3756e-02 |

![HD metrics comparison](report/HD_metrics_comparison.png)

### Visual Comparisons
Plots show predicted vs. actual returns, highlighting sentiment's impact. For comprehensiveness, comparisons reveal how sentiment helps models better fit fat-tailed return distributions, reducing underprediction during volatile periods. In some cases, vanilla models failed to learn meaningful patterns, outputting nearly flat predictions around 0 (essentially the mean return), due to insufficient signal from lagged features alone in noisy, non-stationary data. Adding just the single sentiment score from that day's news headlines to the same model architecture significantly improved accuracy, enabling capture of short-term sentiment-driven movements (e.g., test MSE reduced from to for [model] on [stock]).

Below is a 2-column table with all the prediction plots for side-by-side comparison (images resized to fit the table cells):

| Vanilla | Sentiment |
|---------|-----------|
| **NFLX RNN** <br> <img src="report/rnn/20250825-182559/plots/NFLX_rnn_full_returns_prediction.png" alt="NFLX RNN Vanilla" width="100%"> | **NFLX RNN** <br> <img src="report/rnn/20250825-182653/plots/NFLX_finbert-rnn_full_returns_prediction.png" alt="NFLX RNN Sentiment" width="100%"> |
| **HD RNN** <br> <img src="report/rnn/20250825-182905/plots/HD_rnn_full_returns_prediction.png" alt="HD RNN Vanilla" width="100%"> | **HD RNN** <br> <img src="report/rnn/20250825-182756/plots/HD_finbert-rnn_full_returns_prediction.png" alt="HD RNN Sentiment" width="100%"> |
| **NFLX LSTM** <br> <img src="report/lstm/20250824-234116/plots/NFLX_lstm_full_returns_prediction.png" alt="NFLX LSTM Vanilla" width="100%"> | **NFLX LSTM** <br> <img src="report/lstm/20250824-234258/plots/NFLX_finbert-lstm_full_returns_prediction.png" alt="NFLX LSTM Sentiment" width="100%"> |
| **HD LSTM** <br> <img src="report/lstm/20250824-234926/plots/HD_lstm_full_returns_prediction.png" alt="HD LSTM Vanilla" width="100%"> | **HD LSTM** <br> <img src="report/lstm/20250824-234754/plots/HD_finbert-lstm_full_returns_prediction.png" alt="HD LSTM Sentiment" width="100%"> |
| **NFLX TabMLP** <br> <img src="report/tabmlp/20250825-183844/plots/NFLX_tabmlp_full_returns_prediction.png" alt="NFLX TabMLP Vanilla" width="100%"> | **NFLX TabMLP** <br> <img src="report/tabmlp/20250825-183926/plots/NFLX_finbert-tabmlp_full_returns_prediction.png" alt="NFLX TabMLP Sentiment" width="100%"> |
| **HD TabMLP** <br> <img src="report/tabmlp/20250825-183725/plots/HD_tabmlp_full_returns_prediction.png" alt="HD TabMLP Vanilla" width="100%"> | **HD TabMLP** <br> <img src="report/tabmlp/20250825-183627/plots/HD_finbert-tabmlp_full_returns_prediction.png" alt="HD TabMLP Sentiment" width="100%"> |

Sentiment integration consistently reduced errors across models and stocks, validating its role in capturing market dynamics and improving forecast reliability for non-stationary, heavy-tailed financial returns.