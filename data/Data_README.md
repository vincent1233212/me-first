- Date range: 1990-01-02 to 2025-10-30
- Trading days: 9025
- 50 US stocks from S&P 500, balanced across 11 GICS sectors
- Data sources: Yahoo Finance (equities, VIX), FRED (DGS10, GDP, CPI, UNRATE)

Files
- `stock_list_v2.csv`: 50 stocks with sector classification
- `prices_adj_close_v2.csv`: Adjusted close prices (wide format)
- `log_returns_v2.csv`: Daily log returns for 50 stocks + S&P 500 + VIX
- `market_loss_v2.csv`: Daily losses (L = -r) for 50 stocks + average market loss
- `macro_monthly.csv`: Monthly GDP growth, CPI YoY, unemployment rate
- `dgs10.csv`: Daily US 10Y Treasury yield
- `final_dataset_v2.csv`: Merged daily dataset (58 columns)

 Data Processing
- Master calendar: S&P 500 trading days
- Prices forward-filled up to 5 days for missing values
- Log returns: r_t = ln(P_t / P_{t-1})
- Market loss: L_t = -r_t
- Macro variables: monthly series forward-filled to daily frequency
- GDP: quarterly series forward-filled to monthly, then daily (limitation: look-ahead bias)

Known Limitations
1. **Survivorship bias**: 50 stocks are current S&P 500 constituents; delisted stocks not included.
2. **Backfilled data**: Some stocks (e.g., BRO, COHR, CRH) had backfilled data in early 1990s; these were replaced.
3. **ESS and KIM**: ESS IPO 1994-06-07, KIM IPO 1991-11-22; NaN before these dates. Rolling PCA should start from 1994-06-07.
4. **Zero returns**: Some stocks (e.g., TFC 520, PCG 509) have >500 zero log returns due to stale prices or low volatility. This may affect rolling correlation slightly.
5. **Macro forward-fill**: GDP/CPI/unemployment forward-filled to daily; 21 NaN at the beginning of the sample (1990) due to alignment.
6. **VIX**: 1 NaN on 1990-01-02 (first trading day).

 Extreme Events (for EVT)
- Each stock has ~452 extreme downside events (below 5th percentile), sufficient for GPD fitting.
- ESS: 396, KIM: 428 (fewer due to NaN).
