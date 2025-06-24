import requests
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import minimize

def get_all_coins(api_key):
    """
    Fetches all available cryptocurrencies from the CoinGecko Pro API using paginated requests.

    This function retrieves market data for all listed coins, ordered by 24-hour trading volume in descending order (Args = 'order': 'volume_desc').
    It sends multiple requests to the `/coins/markets` endpoint, fetching 250 coins per page (the maximum allowed),
    and continues fetching pages until no more data is returned.

    Args:
        api_key (str): Your CoinGecko Pro or Demo API key.

    Returns:
        list: A list of dictionaries, each representing a coin with market data such as current price,
              market cap, volume, and other available metrics.

    Notes:
        - Respects CoinGecko's rate limit by pausing for 1 second between requests.
        - The function stops automatically when an empty page is returned.
        - Ensure your API plan supports the number of requests you're making.
        - Common fields returned per coin: 'id', 'symbol', 'name', 'current_price', 'market_cap', 'total_volume', etc.
    """

    all_coins = []
    page = 1
    per_page = 250  # Max per page allowed by CoinGecko Pro

    url = 'https://pro-api.coingecko.com/api/v3/coins/markets'
    headers = {'X-CG-Pro-API-Key': api_key}

    while len(all_coins) <= 250:
        params = {
            'vs_currency': 'usd',
            'order': 'volume_desc',
            'per_page': per_page,
            'page': page
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            coins = response.json()
            if not coins:
                break
            all_coins.extend(coins)
            print(f"Fetched page {page}, total coins so far: {len(all_coins)}")
            page += 1
        else:
            print(
                f"Error fetching coin list: {response.status_code}, {response.text}")
            break

        time.sleep(1)

    return all_coins


def retrieve_coin_history(coin_info, start_dt, end_dt, interval, api_key):
    """
    Fetches historical price and volume data for a specified cryptocurrency over a given time range.

    Args:
        coin_info (dict): Dictionary containing coin metadata (must include 'id' and 'symbol').
        start_dt (datetime): Start date for the historical range.
        end_dt (datetime): End date for the historical range.
        interval (str): Data interval (e.g., 'hourly', 'daily').
        api_key (str): CoinGecko Pro API key.

    Returns:
        tuple: (coin symbol, DataFrame with date-indexed price and volume data) or (None, None) on failure.
    """
    coin_id = coin_info.get('id')
    symbol = coin_info.get('symbol', '').upper()
    print(f"Retrieving data for {symbol} ({coin_id})...")

    endpoint = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
        "interval": interval
    }
    headers = {
        "X-CG-Pro-API-Key": api_key
    }

    try:
        res = requests.get(endpoint, headers=headers, params=params)
        if res.status_code == 200:
            result = res.json()
            price_data = result.get("prices", [])
            volume_data = result.get("total_volumes", [])

            if price_data and volume_data:
                df = pd.DataFrame(price_data, columns=["timestamp", "price"])
                df["volume"] = pd.DataFrame(
                    volume_data, columns=["timestamp", "volume"])["volume"]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("date", inplace=True)
                df.drop(columns=["timestamp"], inplace=True)
                return symbol, df
            else:
                print(f"No data found for {symbol}")
        else:
            print(
                f"API error for {symbol} ({coin_id}): {res.status_code}, {res.text}")
    except Exception as error:
        print(f"Exception while fetching {symbol} ({coin_id}): {error}")

    return None, None


def save_coin_data(data, start_date, end_date):
    filename = "data/coin_px_vol_{}_to_{}.pkl".format(
        str(start_date.date()),
        str(end_date.date())
    )

    with open(filename, "wb") as f:
        pickle.dump(data, f)

def compute_performance_metrics(strategy_returns, benchmark_returns=None, rf=0.0):
    strategy_returns = pd.Series(strategy_returns).dropna()

    if strategy_returns.empty:
        return pd.Series({
            'Holding Period (Days)': 0,
            'Cumulative Return': np.nan,
            'Annualized Return': np.nan,
            'Annualized Volatility': np.nan,
            'Sharpe Ratio': np.nan,
            'Sortino Ratio': np.nan,
            'Max Drawdown': np.nan,
            'Win Rate': np.nan,
            'Avg Win': np.nan,
            'Avg Loss': np.nan,
            'Profit Factor': np.nan,
            'Alpha': np.nan,
            'Beta': np.nan,
            'R-squared': np.nan
        })

    metrics = {}

    # Holding Period
    days_held = len(strategy_returns)
    metrics['Holding Period (Days)'] = days_held
    metrics['Holding Period (Months)'] = round(days_held / 21, 2)
    metrics['Holding Period (Years)'] = round(days_held / 252, 2)

    # Cumulative Return
    cumulative = (1 + strategy_returns).prod() - 1
    metrics['Cumulative Return'] = cumulative

    # Annualized Return
    ann_return = (1 + cumulative)**(252 / days_held) - 1
    metrics['Annualized Return'] = ann_return

    # Volatility
    ann_volatility = strategy_returns.std() * np.sqrt(252)
    metrics['Annualized Volatility'] = ann_volatility

    # Sharpe Ratio
    sharpe = (strategy_returns.mean() - rf / 252) / strategy_returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = sharpe

    # Sortino Ratio (using only downside std)
    downside = strategy_returns[strategy_returns < 0]
    if not downside.empty:
        sortino = (strategy_returns.mean() - rf / 252) / downside.std() * np.sqrt(252)
    else:
        sortino = np.nan
    metrics['Sortino Ratio'] = sortino

    # Max Drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    metrics['Max Drawdown'] = drawdown.min()

    # Win Rate, Avg Win/Loss, Profit Factor
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    metrics['Win Rate'] = len(wins) / len(strategy_returns)
    metrics['Avg Win'] = wins.mean() if not wins.empty else 0
    metrics['Avg Loss'] = losses.mean() if not losses.empty else 0
    metrics['Profit Factor'] = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # Alpha / Beta / R^2
    if benchmark_returns is not None:
        benchmark_returns = pd.Series(benchmark_returns).dropna()
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strat = strategy_returns.loc[common_index]
        bench = benchmark_returns.loc[common_index]

        if not strat.empty:
            slope, intercept, r_value, _, _ = stats.linregress(bench, strat)
            metrics['Alpha'] = intercept * 252
            metrics['Beta'] = slope
            metrics['R-squared'] = r_value**2
        else:
            metrics['Alpha'] = np.nan
            metrics['Beta'] = np.nan
            metrics['R-squared'] = np.nan

    return pd.Series(metrics)

def are_too_similar(sym1, sym2):
    return sym1 in sym2 or sym2 in sym1

def get_cointegrated_pairs_stats(asset_combinations, prices_standardized, split_index):
    cointegrated_pairs = []

    for comb in tqdm(asset_combinations, desc="\tFinding cointegrated pairs for Split {}".format(split_index+1)):
        pairs_df = pd.concat(
            [
                prices_standardized[[comb[0]]], 
                prices_standardized[[comb[1]]]
            ], 
            axis=1
        ).dropna()

        if pairs_df.shape[0]>0:
            model = sm.OLS(
                pairs_df[comb[0]], 
                sm.add_constant(pairs_df[comb[1]])
            ).fit()

            residuals = model.resid

            test_statistic = adfuller(residuals)[0]

            adf_pval = adfuller(residuals)[1]

            if adf_pval < 0.05:
                cointegrated_pairs.append(
                    [
                        comb[0],
                        comb[1],
                        model.params[1],
                        float(adf_pval),
                        test_statistic,
                        residuals
                    ]
                )
        else:
            pass   

    # Remove pairs where token names share large substrings
    filtered_pairs = [
        pair for pair in cointegrated_pairs
        if not are_too_similar(pair[0], pair[1])
    ]

    df_cointegration = pd.DataFrame(
        [i[:5] for i in filtered_pairs], 
        columns=[
            "Asset 1", 
            "Asset 2", 
            "beta",
            "ADF p-value", 
            "ADF Test_statistic"
        ]
    )

    df_cointegration["ADF Test_statistic Rank"] = df_cointegration.groupby(
        ["Asset 1"]
    )["ADF Test_statistic"].rank(ascending=False)

    df_cointegration = df_cointegration.loc[
        df_cointegration.groupby(
            ["Asset 1"]
        )['ADF Test_statistic Rank'].idxmax()
    ].reset_index(drop=True)

    return df_cointegration


def agg_pair_returns_signals(df_cointegration, scaler, test_set, is_train, window):
    pairs_signals_df = {}
    pairs_returns_df = {}

    for i in range(df_cointegration.shape[0]):
        coin1, coin2, beta = df_cointegration.loc[i, :].tolist()[:3]

        if is_train==False:
            test_set_standardized = pd.DataFrame(
                data=scaler.transform(X=test_set),
                columns=test_set.columns,
                index=test_set.index
                ).fillna(method='ffill')
        
        else:
            test_set_standardized = test_set

        spread = (test_set_standardized[coin1] - beta * test_set_standardized[coin2]).dropna()

        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std

        signals = pd.Series(index=zscore.index, data=0)

        # Entry signals
        signals[zscore > 1] = -1  # Short Spread
        signals[zscore < -1] = 1  # Long Spread

        # Exit signals
        signals[(zscore > -0.5) & (zscore < 0.5)] = 0

        # Forward fill signals
        position = signals.ffill().fillna(0)
  
        # Returns (from your already computed returns)
        ret = test_set.pct_change()[coin1] - beta * test_set.pct_change()[coin2]
        strategy_returns = position.shift(1) * ret

        pairs_signals_df["-".join([coin1, coin2])] = signals.tolist()
        pairs_returns_df["-".join([coin1, coin2])] = strategy_returns.tolist()

    # pairs_returns_df = pd.DataFrame(pairs_returns_df)

    # pairs_signals_df = pd.DataFrame(pairs_signals_df)

    return pairs_returns_df, pairs_signals_df