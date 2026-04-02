"""
Run the walk-forward backtest.

Usage:
  python scripts/backtest.py --days 180 --window 30

This fetches ETHUSDT 15m candles from Binance (mainnet, no auth needed for
historical data) and runs the backtest across non-overlapping 30-day windows.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

from binance.client import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crypto_bot.backtesting.backtester import Backtester
from crypto_bot.core.regime_detector import RegimeDetector
from crypto_bot.data.data_collector import DataCollector
from crypto_bot.signals.signal_generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    ]
)

os.makedirs('logs', exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=180, help='Total days of history to fetch')
    parser.add_argument('--window', type=int, default=30, help='Walk-forward window size in days')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--save-csv', action='store_true', help='Save trades to CSV')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    regime_cfg = config.get('regime', {})
    signal_cfg = config.get('signal', {})

    # Binance client (no API key needed for historical data)
    client = Client()
    collector = DataCollector(client)

    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%-d %b, %Y')
    df = collector.fetch_klines(
        symbol=config['symbol'],
        interval=config['interval'],
        start_date=start_date,
    )

    regime_detector = RegimeDetector(
        adx_period=regime_cfg.get('adx_period', 14),
        adx_trend_threshold=regime_cfg.get('adx_trend_threshold', 25.0),
        adx_range_threshold=regime_cfg.get('adx_range_threshold', 20.0),
        ema_slope_period=regime_cfg.get('ema_slope_period', 50),
        ema_slope_lookback=regime_cfg.get('ema_slope_lookback', 5),
    )

    signal_gen = SignalGenerator(
        atr_sl_multiplier=signal_cfg.get('atr_sl_multiplier', 1.0),
        atr_tp_multiplier=signal_cfg.get('atr_tp_multiplier', 3.0),
        min_rr=signal_cfg.get('min_rr', 1.8),
        rsi_oversold=signal_cfg.get('rsi_oversold', 30.0),
        rsi_overbought=signal_cfg.get('rsi_overbought', 70.0),
        ema_fast=signal_cfg.get('ema_fast', 9),
        ema_slow=signal_cfg.get('ema_slow', 21),
        pullback_tolerance=signal_cfg.get('pullback_tolerance', 0.005),
        rsi_pullback_min=signal_cfg.get('rsi_pullback_min', 40.0),
        rsi_pullback_max=signal_cfg.get('rsi_pullback_max', 60.0),
        adx_max=signal_cfg.get('adx_max', 40.0),
    )

    backtester = Backtester(
        regime_detector=regime_detector,
        signal_generator=signal_gen,
        window_days=args.window,
        leverage=config.get('leverage', 2),
    )

    results = backtester.run(df)

    if args.save_csv:
        trades_df = backtester.to_dataframe(results)
        if not trades_df.empty:
            csv_path = f'logs/backtest_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            trades_df.to_csv(csv_path, index=False)
            logging.info(f"Trades saved to {csv_path}")


if __name__ == '__main__':
    main()
