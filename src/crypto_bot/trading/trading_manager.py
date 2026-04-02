"""
Trading Manager

Main loop:
1. Fetch latest candles from Binance
2. Detect market regime (RegimeDetector)
3. Generate signal (SignalGenerator)
4. Open/close positions via OrdersManager
5. Monitor open position against dynamic TP/SL from the signal
"""

import logging
import time

import pandas as pd
from binance.client import Client

from crypto_bot.core.regime_detector import RegimeDetector
from crypto_bot.data.data_collector import DataCollector
from crypto_bot.orders.models.position_information import TAKER_FEE_RATE, MAKER_FEE_RATE
from crypto_bot.orders.orders_manager import OrdersManager
from crypto_bot.signals.signal_generator import SignalGenerator, SignalType
from crypto_bot.trading.trading_status import TradingStatus


class TradingManager:
    def __init__(
        self,
        binance_client: Client,
        config: dict,
    ):
        self.config = config
        self.orders = OrdersManager(binance_client)
        self.data = DataCollector(binance_client)
        self.regime_detector = RegimeDetector(
            adx_period=config.get('adx_period', 14),
            adx_trend_threshold=config.get('adx_trend_threshold', 25.0),
            adx_range_threshold=config.get('adx_range_threshold', 20.0),
        )
        self.signal_gen = SignalGenerator(
            atr_sl_multiplier=config.get('atr_sl_multiplier', 1.0),
            atr_tp_multiplier=config.get('atr_tp_multiplier', 2.0),
            min_rr=config.get('min_rr', 1.8),
            rsi_oversold=config.get('rsi_oversold', 30.0),
            rsi_overbought=config.get('rsi_overbought', 70.0),
        )

        self.symbol = config['symbol']
        self.interval = config['interval']
        self.leverage = config['leverage']
        self.investment_usdt = config['investment_usdt']
        self.loop_interval_secs = config.get('loop_interval_secs', 60)
        self.candles_lookback = config.get('candles_lookback', 300)

        # Track the current signal's TP/SL so we can monitor them each loop
        self.current_signal = None
        self.status = TradingStatus()

    def run(self, max_iterations: int = -1):
        logging.info(f"Starting TradingManager: {self.symbol} {self.interval} leverage={self.leverage}x")

        self.orders.futures_change_leverage(self.symbol, self.leverage)

        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 5

        while max_iterations == -1 or iteration < max_iterations:
            try:
                self._cycle()
                consecutive_errors = 0
                iteration += 1
                logging.info(f"[Iteration {iteration}] Balance: {self.status.get_balance():+.3f} USDT | "
                             f"Closed trades: {len(self.status.closed_positions)}")
                logging.info(f"Sleeping {self.loop_interval_secs}s until next cycle...")
                time.sleep(self.loop_interval_secs)

            except Exception as e:
                consecutive_errors += 1
                logging.exception(f"Error in trading cycle ({consecutive_errors}/{max_consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logging.critical("Too many consecutive errors — closing positions and stopping")
                    self._emergency_close()
                    break
                time.sleep(10)

        logging.info(f"Bot stopped. Final balance: {self.status.get_balance():+.3f} | "
                     f"Trades: {len(self.status.closed_positions)}")
        return self.status

    def _cycle(self):
        # 1. Fetch recent candles
        df = self._fetch_candles()

        # 2. Detect regime + enrich df with ADX, ATR, EMA slope
        df = self.regime_detector.add_regime_to_df(df)
        regime = self.regime_detector.get_current_regime(df)

        # 3. Generate signal
        signal = self.signal_gen.generate(df, regime)
        logging.info(f"Signal: {signal.signal.value} | Regime: {signal.regime.value} | {signal.reason}")

        # 4. Manage open position (check TP/SL)
        if self.orders.is_there_an_open_position(self.symbol):
            self._manage_open_position(df)
            return  # Don't open new positions while one is active

        # 5. Open new position if signal is actionable
        if signal.signal == SignalType.HOLD:
            logging.info("HOLD — no trade this cycle")
            return

        current_price = self.orders.get_current_price(self.symbol)
        qty = OrdersManager.qty_from_usdt(current_price, self.investment_usdt)

        if signal.signal == SignalType.LONG:
            order = self.orders.create_long_order(self.symbol, qty)
        else:
            order = self.orders.create_short_order(self.symbol, qty)

        time.sleep(2)
        position_info = self.orders.get_current_position_info(self.symbol)
        if position_info and position_info.positionAmt != 0:
            self.status.open_position(position_info)
            self.current_signal = signal
            logging.info(
                f"Position opened: {signal.signal.value.upper()} "
                f"entry={position_info.entryPrice:.2f} "
                f"TP={signal.take_profit:.2f} SL={signal.stop_loss:.2f} "
                f"R:R={signal.risk_reward:.2f}"
            )
        else:
            logging.warning("Order placed but position not confirmed yet")

    def _manage_open_position(self, df: pd.DataFrame):
        """Check TP/SL against current market price."""
        position_info = self.orders.get_current_position_info(self.symbol)
        if position_info is None or position_info.positionAmt == 0:
            if self.status.has_open_position():
                # Position was closed externally (e.g. liquidation)
                logging.warning("Position disappeared — clearing tracking")
                self.status.current_position = None
                self.current_signal = None
            return

        # Update tracking
        current_price = float(df.iloc[-1]['close'])
        self.status.status_update(current_price)

        if self.current_signal is None:
            logging.warning("Open position with no tracked signal — skipping TP/SL check")
            return

        is_long = position_info.positionAmt > 0
        tp = self.current_signal.take_profit
        sl = self.current_signal.stop_loss

        hit_tp = (is_long and current_price >= tp) or (not is_long and current_price <= tp)
        hit_sl = (is_long and current_price <= sl) or (not is_long and current_price >= sl)

        net_pnl = self._net_pnl_pct(position_info)
        pos_type = "LONG" if is_long else "SHORT"
        logging.info(
            f"Monitoring {pos_type}: price={current_price:.2f} TP={tp:.2f} SL={sl:.2f} "
            f"net_pnl={net_pnl:+.2%}"
        )

        if hit_tp:
            logging.info(f"TAKE PROFIT hit at {current_price:.2f}")
            self._close_position(position_info)
        elif hit_sl:
            logging.info(f"STOP LOSS hit at {current_price:.2f}")
            self._close_position(position_info)

    def _close_position(self, position_info):
        self.orders.close_current_positions(self.symbol)
        self.status.close_position(position_info)
        self.current_signal = None
        logging.info(f"Position closed. Accumulated PNL: {self.status.accumulated_profit:+.3f} USDT")

    def _emergency_close(self):
        try:
            self.orders.close_current_positions(self.symbol)
        except Exception as e:
            logging.error(f"Emergency close failed: {e}")

    def _fetch_candles(self) -> pd.DataFrame:
        """Fetch the last N candles from Binance REST API."""
        klines = self.data.client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.candles_lookback
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    def _net_pnl_pct(position_info) -> float:
        from crypto_bot.orders.models.position_information import TraderPosition
        return TraderPosition.calculate_unrealized_pnl(position_info, include_fees=True)
