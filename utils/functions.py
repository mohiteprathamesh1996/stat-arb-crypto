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
