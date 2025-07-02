import requests  # type: ignore
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # type: ignore
from tqdm import tqdm  # type: ignore
import statsmodels.api as sm  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore


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
                f"Error fetching coin list: {
                    response.status_code}, {
                    response.text}")
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
                f"API error for {symbol} ({coin_id}): {
                    res.status_code}, {
                    res.text}")
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


def split_dataset_by_ratio(
        price_df,
        train_ratio=0.6,
        val_ratio=0.25,
        test_ratio=0.15):
    """
    Splits a time-indexed DataFrame into train, validation, and test sets based on date ranges.

    Args:
        price_df (pd.DataFrame): DataFrame with datetime index.
        train_ratio (float): Portion for training (default 60%)
        val_ratio (float): Portion for validation (default 25%)
        test_ratio (float): Portion for out-of-sample testing (default 15%)

    Returns:
        train_df, val_df, test_df: Split DataFrames
    """

    # Ensure the index is datetime and sorted
    df = price_df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a datetime index")

    # Calculate the total time span
    start_date = df.index.min()
    end_date = df.index.max()
    total_days = (end_date - start_date).days

    # Compute split durations
    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)

    # Compute actual split dates
    train_end = start_date + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=val_days)

    # Slice the DataFrame
    train_df = df.loc[:train_end]
    val_df = df.loc[train_end + pd.Timedelta(days=1):val_end]
    test_df = df.loc[val_end + pd.Timedelta(days=1):]

    # Summary print
    print(
        f"Date range: {
            start_date.date()} → {
            end_date.date()}  ({total_days} days)")
    print(
        f"Training:   {
            train_df.index.min().date()} → {
            train_df.index.max().date()} ({
                len(train_df)} days)")
    print(
        f"Validation: {
            val_df.index.min().date()} → {
            val_df.index.max().date()} ({
                len(val_df)} days)")
    print(
        f"Out-of-Sample Test: {
            test_df.index.min().date()} → {
            test_df.index.max().date()} ({
                len(test_df)} days)")

    return train_df, val_df, test_df


def ols_fit(data, coin1, coin2):
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller

    df = data[[coin1, coin2]].dropna()

    X = sm.add_constant(df[coin1])
    y = df[coin2]

    model = sm.OLS(y, X).fit()
    adf_residual_stationarity = adfuller(
        x=model.resid
    )

    results = {
        "coin1": coin1,
        "coin2": coin2,
        "alpha": model.params.iloc[0],
        "beta": model.params.iloc[1],
        "adf_test_statistic": adf_residual_stationarity[0],
        "adf_p_value": adf_residual_stationarity[1]
    }

    return pd.Series(results)


def get_cointegrated_pairs(prices):
    from itertools import combinations
    import re

    pairs = pd.DataFrame(
        list(
            combinations(
                iterable=prices.columns,
                r=2
            )
        ),
        columns=["coin1", "coin2"]
    )

    pairs["str_match"] = pairs.apply(
        lambda x: len(
            re.findall(re.escape(x['coin1']), x['coin2']) or
            re.findall(re.escape(x['coin2']), x['coin1'])),
        axis=1
    )

    pairs = pairs.query(
        expr="str_match < 1"
    ).reset_index(
        drop=True
    ).drop(
        columns=["str_match"]
    )

    pairs["correlation"] = pairs.apply(
        lambda row: np.corrcoef(
            prices[row["coin1"]],
            prices[row["coin2"]])[0, 1],
        axis=1
    ).dropna()

    pairs = pairs[
        # pairs["correlation"].abs()>0.95
        pairs["correlation"] > 0.96
    ].apply(
        lambda row: ols_fit(
            data=prices,
            coin1=row['coin1'],
            coin2=row['coin2']
        ),
        axis=1
    ).query(
        expr="adf_p_value < 0.01"
    ).reset_index(drop=True)

    pairs = pairs.loc[
        pairs.loc[
            pairs.groupby('coin1')['adf_test_statistic'].idxmax()
        ].groupby('coin2')['adf_test_statistic'].idxmax()
    ].reset_index(drop=True)

    final_pairs = pairs[["coin1", "coin2"]].values

    return final_pairs


def comparison_absolute_vs_norm_prices(pair, train, train_norm, ax1, ax2):
    # Plot raw prices
    train[pair].plot(ax=ax1)
    ax1.set_title(f"{pair[0]} & {pair[1]} - Absolute Prices")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    # Plot normalized prices
    train_norm[pair].plot(ax=ax2)
    ax2.set_title(
        f"{pair[0]} & {pair[1]} - Normalized Prices (Zero Mean, Unit Variance)")
    ax2.set_ylabel("Normalized Price")
    ax2.grid(True)


def get_rolling_signals(prices, select_pairs, rolling_window=90):
    rolling_params = {}

    for pair in select_pairs:
        coin1, coin2 = pair

        rolling_beta = (
            prices[coin1].rolling(window=rolling_window).cov(prices[coin2]) /
            prices[coin1].rolling(window=rolling_window).var()
        )

        rolling_alpha = (
            prices[coin2].rolling(window=rolling_window).mean() -
            (rolling_beta * prices[coin1].rolling(window=rolling_window).mean())
        )

        rolling_spread = (
            prices[coin1] -
            (
                rolling_beta * prices[coin2] +
                rolling_alpha
            )
        )

        rolling_z_score = (
            rolling_spread - rolling_spread.rolling(window=rolling_window).mean()
        ) / rolling_spread.rolling(window=rolling_window).std()

        rolling_params[(coin1, coin2)] = {
            "rolling_alpha": rolling_alpha,
            "rolling_beta": rolling_beta,
            "rolling_spread": rolling_spread,
            "rolling_z_score": rolling_z_score
        }

    return rolling_params


def get_long_short_signals(
        prices,
        select_pairs,
        rolling_params,
        entry_threshold=1,
        exit_threshold=0.5):
    from itertools import chain

    in_sample_positions = pd.DataFrame(
        index=prices.index,
        columns=list(set(chain.from_iterable(select_pairs)))
    ).fillna(0)

    for pair in select_pairs:
        coin1, coin2 = pair

        # betas = pd.DataFrame(rolling_params[coin1, coin2]).loc[prices.index]["rolling_beta"]
        zscores = pd.DataFrame(
            rolling_params[coin1, coin2]).loc[prices.index]["rolling_z_score"]

        in_sample_positions.loc[zscores > entry_threshold, coin1] = -1
        in_sample_positions.loc[zscores < -entry_threshold, coin1] = +1
        in_sample_positions.loc[abs(zscores) <= exit_threshold, coin1] = 0

        in_sample_positions.loc[zscores > entry_threshold, coin2] = +1
        in_sample_positions.loc[zscores < -entry_threshold, coin2] = -1
        in_sample_positions.loc[abs(zscores) <= exit_threshold, coin2] = 0

    in_sample_positions.ffill(inplace=True)

    # in_sample_positions = in_sample_positions.divide(
    #     in_sample_positions.abs().sum(axis=1), axis=0
    # ).fillna(0)

    return in_sample_positions


def compute_performance_metrics(
        strategy_returns,
        benchmark_returns=None,
        rf=0.0):
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
    sharpe = (strategy_returns.mean() - rf / 252) / \
        strategy_returns.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = sharpe

    # Sortino Ratio (using only downside std)
    downside = strategy_returns[strategy_returns < 0]
    if not downside.empty:
        sortino = (strategy_returns.mean() - rf / 252) / \
            downside.std() * np.sqrt(252)
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
    metrics['Profit Factor'] = wins.sum() / abs(losses.sum()
                                                ) if losses.sum() != 0 else np.nan

    # Alpha / Beta / R^2
    if benchmark_returns is not None:
        benchmark_returns = pd.Series(benchmark_returns).dropna()
        common_index = strategy_returns.index.intersection(
            benchmark_returns.index)
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


def create_surface_plots(df, values, cmap="viridis"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    n = len(values)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

    for i, value in enumerate(values):
        # Create a 2D grid of z values (e.g., Alpha)
        z_grid = df.pivot(
            index="Exit Threshold",
            columns="Entry Threshold",
            values=value
        )

        # Convert index and columns to meshgrid
        x = z_grid.columns.values
        y = z_grid.index.values
        X, Y = np.meshgrid(x, y)
        Z = z_grid.values

        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9)

        ax.set_xlabel('Entry Threshold')
        ax.set_ylabel('Exit Threshold')
        ax.set_zlabel(value)
        ax.set_title(f'{value} vs Entry/Exit Thresholds')

        fig.colorbar(surf, ax=ax, shrink=0, aspect=1)

    # plt.suptitle("Evaluating Strategy Sensitivity to Entry and Exit Thresholds\n\n")
    plt.tight_layout()
    plt.show()
