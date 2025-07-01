# Statistical Arbitrage in Cryptocurrencies: A Mean-Reverting Systematic Pairs Trading Framework

This repository contains the full code and research supporting the application and demonstration of a fully systematic, market-neutral long/short strategy using historical crypto price data and cointegration-based signal generation.



## Overview

This project applies a classic **mean-reversion pairs trading strategy** to the volatile cryptocurrency market using a rigorous quant research pipeline:

-  Data collection from Binance API  
-  Cointegration detection via Engle-Granger test  
-  Rolling estimation of alpha, beta, spread, and z-score  
-  Signal generation based on dynamic z-score thresholds  
-  Grid search optimization for parameter tuning  
-  Robust out-of-sample backtesting  
-  Visualization of trade signals, performance surfaces, and benchmark comparison  
-  Evaluation using professional-grade metrics (Sharpe, Alpha, Max Drawdown, etc.)

---

## 🔧 Tech Stack

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- `matplotlib`, `seaborn`, `plotly`
- `yfinance` / `ccxt` or `Binance API` (for live data)
- `autopep8` (for formatting)

---

## 📁 File Structure

```bash
crypto-pairs-trading/
├── data/                          # Raw & processed CSV data (training, validation, test)
├── src/
│   ├── data_loader.py             # Fetches and prepares historical crypto prices
│   ├── cointegration.py          # Engle-Granger test and pair selection
│   ├── signal_generation.py      # Rolling alpha, beta, spread, and z-score calculation
│   ├── backtest.py               # Core vectorized backtesting logic
│   ├── optimization.py           # Grid search over entry/exit thresholds
│   └── performance_metrics.py    # Metrics like Sharpe, Alpha, MDD, etc.
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_cointegration_analysis.ipynb
│   ├── 03_signal_generation.ipynb
│   ├── 04_backtest_engine.ipynb
│   └── 05_validation_test_analysis.ipynb
│
├── plots/                        # All exported plots: performance charts, trade signals
├── results/                      # Final evaluation results & parameter surfaces
├── main.py                       # End-to-end execution pipeline
├── README.md                     # Project documentation
└── requirements.txt              # All Python dependencies
