import os
from datetime import datetime
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from utils.functions import get_all_coins, retrieve_coin_history, save_coin_data

load_dotenv()

api_key = os.getenv("PM_WSQ_PROJECT_API")
start_dt = pd.Timestamp.now() - pd.DateOffset(years=3)
end_dt = pd.Timestamp.now()
interval = "daily"


all_coins = get_all_coins(api_key=api_key)

coin_price_volume = {}

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(
            retrieve_coin_history,
            coin,
            start_dt,
            end_dt,
            interval,
            api_key
        )
        for coin in all_coins
    ]

    for future in as_completed(futures):
        try:
            coin_symbol, df = future.result()
            if coin_symbol and df is not None:
                coin_price_volume[coin_symbol] = df
        except Exception as e:
            print(f"Error in future: {e}")

        time.sleep(5)

save_coin_data(
    data=coin_price_volume,
    start_date=start_dt,
    end_date=end_dt
)
