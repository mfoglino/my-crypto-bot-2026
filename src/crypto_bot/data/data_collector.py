import logging
import pandas as pd
from binance.client import Client


class DataCollector:
    def __init__(self, client: Client):
        self.client = client

    def fetch_klines(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        logging.info(f"Fetching {symbol} {interval} candles from {start_date} to {end_date or 'now'}")

        klines = self.client.get_historical_klines(
            symbol=symbol, interval=interval,
            start_str=start_date, end_str=end_date
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df[['open', 'high', 'low', 'close', 'volume']]

        logging.info(f"Fetched {len(df)} candles: {df.index[0]} → {df.index[-1]}")
        return df

    def save_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path)
        logging.info(f"Saved {len(df)} rows to {path}")

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        logging.info(f"Loaded {len(df)} rows from {path}")
        return df
