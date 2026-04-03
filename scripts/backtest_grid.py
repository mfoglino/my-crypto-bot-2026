"""
Grid Bot Backtest

Usage:
  python scripts/backtest_grid.py --days 365
  python scripts/backtest_grid.py --days 180 --config config/grid_config.json --save-csv
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

from binance.client import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crypto_bot.data.data_collector import DataCollector
from crypto_bot.grid.grid_backtester import GridBacktester
from crypto_bot.grid.grid_config import GridConfig

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/backtest_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--config", type=str, default="config/grid_config.json")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = json.load(f)

    config = GridConfig.from_dict(raw)

    client = Client()  # No auth needed for historical data
    collector = DataCollector(client)

    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%-d %b, %Y")
    df = collector.fetch_klines(
        symbol=config.symbol,
        interval="1h",
        start_date=start_date,
    )

    backtester = GridBacktester()
    result = backtester.run(df, config)

    print(result.summary())

    if args.save_csv and result.trades:
        import pandas as pd
        rows = [
            {
                "slot": t.slot,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl_usdt": t.pnl_usdt,
                "pnl_pct": t.pnl_pct,
                "pnl_pct_leveraged": t.pnl_pct_leveraged,
            }
            for t in result.trades
        ]
        path = f"logs/grid_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        logging.info(f"Trades saved to {path}")


if __name__ == "__main__":
    main()
