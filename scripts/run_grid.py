"""
Grid Bot — Live / Testnet

Usage:
  # Testnet
  python scripts/run_grid.py --config config/grid_config_testnet.json --testnet

  # Live (only after validating on testnet)
  python scripts/run_grid.py --config config/grid_config.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from binance.client import Client
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crypto_bot.grid.grid_config import GridConfig
from crypto_bot.grid.grid_manager import GridManager

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/grid_config_testnet.json")
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--poll-secs", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = json.load(f)

    config = GridConfig.from_dict(raw)

    if args.testnet:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            logging.error("Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET in .env")
            sys.exit(1)
        logging.info("Running on TESTNET")
        client = Client(api_key, api_secret, testnet=True)
    else:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logging.error("Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env")
            sys.exit(1)
        logging.warning("Running on MAINNET — real money!")
        client = Client(api_key, api_secret)

    grid = GridManager(client, config)

    try:
        grid.setup()
        grid.run(poll_interval_secs=args.poll_secs)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        grid.teardown()


if __name__ == "__main__":
    main()
