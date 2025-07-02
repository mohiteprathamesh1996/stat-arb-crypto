# Statistical Arbitrage in Cryptocurrencies: A Mean-Reverting Systematic Pairs Trading Framework

This repository contains the complete codebase, research artifacts, and visualizations supporting the development of a fully systematic, market-neutral long/short trading strategy applied to the cryptocurrency market. The methodology leverages classical statistical arbitrage techniques—specifically cointegration-based pairs trading—to exploit mean-reverting behavior across highly liquid crypto assets. This work was conducted as part of an academic project submitted in partial fulfillment of the **Wall Street Quants** bootcamp, with the objective of applying institutional-grade quantitative techniques to a modern digital asset class. The pipeline spans from raw data acquisition through robust model validation, backtesting, threshold optimization, and final performance evaluation. The strategy is entirely coded in Python and follows a clean, reproducible research workflow. 



## Disclaimer

This project is intended solely for educational and research purposes. It does not constitute investment advice, and none of the strategies or analyses presented should be interpreted as recommendations to trade financial instruments. Always conduct your own due diligence and consult a registered financial advisor before making investment decisions

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


## Tech Stack

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- `matplotlib`, `seaborn`, `plotly`
- `ccxt` or `Binance API` (for live data)
- `autopep8` (for formatting)

---

## File Structure

```bash
crypto-pairs-trading/
├── data/                                           # Raw & processed CSV data (training, validation, test)
├── utils/
│   ├── functions.py                                # Reusable code: signal logic, rolling stats, performance metrics 
├── notebooks/
│   ├── Pairs Trading Using Cointegration.ipynb     # Core strategy development and experiments
├── data_pipeline.py                                # End-to-end pipeline to download and preprocess crypto price data
├── visualization.py                                # Custom plots: signal overlays, trade flows, 3D threshold surfaces
├── README.md                                       # Project documentation (this file)
├── requirements.txt                                # Python dependencies
```


## License
This project is licensed under the MIT License.

## Contributors
Prathamesh Mohite – Data Scientist

## Support & Feedback
Have questions, feature requests, or feedback? Let’s connect!

Email: mohite.p@northeastern.edu

[LinkedIn Profile](https://www.linkedin.com/in/prathameshmohite96/)
