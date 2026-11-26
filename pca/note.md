# PCA Indicators Dataset

## 1. Dataset Overview

The `pca_features _(from_1994-07-06)` dataset contains five selected PCA-derived risk indicators.

PCA was performed on the standardized daily log returns of 50 selected US stocks across 11 sectors. Seven principal components were retained based on the Kaiser criterion, explaining approximately 51.91% of the total variance.

The five indicators were selected from 15 initial candidates based on financial interpretation, conditional distribution analysis, ROC-AUC, Average Precision and preliminary chronological validation.

## 2. Description of Indicators

| Indicator | Calculation | Financial Interpretation |
|---|---|---|
| `PC1_vol_5` | 5-day rolling standard deviation of PC1 scores | Short-term systematic market volatility |
| `PC1_vol_20` | 20-day rolling standard deviation of PC1 scores | Persistent systematic market volatility |
| `PC1_sum_20` | Sum of PC1 scores over the previous 20 trading days, including the current day | Cumulative market-factor movement; negative values may indicate market deterioration |
| `T2_mean_5` | 5-day rolling mean of Hotelling's T² statistic, calculated using the seven retained PCs | Persistent abnormal movements within the retained principal component space |
| `SPE` | Sum of squared differences between the original standardized returns and their PCA reconstruction | Abnormal stock movements not explained by the seven retained principal components |

PC1 represents the general market factor, with a training-period correlation of approximately 0.945 with S&P 500 daily returns.

Hotelling's T² and SPE provide complementary measures of market stress: the former measures abnormal movements within the retained PCA space, while the latter measures abnormal movements outside it.

## 3. Important Dates

| Item | Date / Period |
|---|---|
| Original log-return dataset | 1990-01-02 to 2025-10-30 |
| PCA estimation period | 1994-06-08 to 2019-12-31 |
| PCA testing period | 2020-01-01 to 2025-10-30 |
| First complete indicator observation | 1994-07-06 |
| Last indicator observation | 2025-10-30 |
| Expected number of complete observations | 7,885 |

The first 19 PCA observations were excluded from the indicator dataset because the 20-day rolling indicators require sufficient historical observations.
