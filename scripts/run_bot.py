"""
Run the live/testnet bot.

Usage:
  # Testnet
  python scripts/run_bot.py --config config/config_testnet.json --testnet

  # Live (mainnet) — only after validating on testnet
  python scripts/run_bot.py --config config/config.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from binance.client import Client
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crypto_bot.trading.trading_manager import TradingManager

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    ]
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--testnet', action='store_true', help='Use Binance testnet')
    parser.add_argument('--max-iterations', type=int, default=-1, help='Stop after N iterations (-1 = forever)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Flatten regime/signal config into top-level for TradingManager
    merged = {**config, **config.get('regime', {}), **config.get('signal', {})}

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logging.error("Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env")
        sys.exit(1)

    if args.testnet:
        logging.info("Running on TESTNET")
        client = Client(api_key, api_secret, testnet=True)
    else:
        logging.warning("Running on MAINNET — real money!")
        client = Client(api_key, api_secret)

    bot = TradingManager(binance_client=client, config=merged)
    bot.run(max_iterations=args.max_iterations)


if __name__ == '__main__':
    main()
