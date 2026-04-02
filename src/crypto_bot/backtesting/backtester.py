"""
Walk-Forward Backtester

Why walk-forward instead of simple backtesting?
  Simple backtest: train on 2024, test on 2024 → looks great but overfits.
  Walk-forward:    train on Jan, test on Feb → train on Feb, test on Mar → ...
                   Each test period is truly out-of-sample.

This is the only honest way to validate a strategy before risking real money.

The regime+signal approach here has no training phase (pure rule-based),
so walk-forward just means we run the strategy on non-overlapping windows
and report per-window AND aggregate metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from crypto_bot.core.regime_detector import RegimeDetector, Regime
from crypto_bot.signals.signal_generator import SignalGenerator, SignalType, TradingSignal
from crypto_bot.orders.models.position_information import TAKER_FEE_RATE


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str          # 'long' or 'short'
    entry_price: float
    exit_price: float
    take_profit: float
    stop_loss: float
    atr: float
    regime: str
    exit_reason: str        # 'tp', 'sl', 'end_of_data'
    pnl_pct: float         # net % after fees (on invested capital, not leveraged)
    pnl_pct_leveraged: float  # net % with leverage applied


@dataclass
class WindowResult:
    start: str
    end: str
    trades: List[Trade] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades)

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct_leveraged for t in self.trades)

    @property
    def avg_rr(self) -> float:
        if not self.trades:
            return 0.0
        rrs = [(t.take_profit - t.entry_price) / (t.entry_price - t.stop_loss)
               if t.direction == 'long'
               else (t.entry_price - t.take_profit) / (t.stop_loss - t.entry_price)
               for t in self.trades]
        return float(np.mean(rrs))

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.pnl_pct_leveraged for t in self.trades])
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        return float(drawdown.max())


class Backtester:
    """
    Parameters
    ----------
    window_days : int
        Size of each test window in days. Default 30.
    leverage : int
        Leverage used in trades (for PNL calculation). Default 2.
    fee_rate : float
        Fee per side. Default TAKER_FEE_RATE (0.04%).
    """

    def __init__(
        self,
        regime_detector: RegimeDetector = None,
        signal_generator: SignalGenerator = None,
        window_days: int = 30,
        leverage: int = 2,
        fee_rate: float = TAKER_FEE_RATE,
        min_candles_for_signal: int = 100,
    ):
        self.regime_detector = regime_detector or RegimeDetector()
        self.signal_gen = signal_generator or SignalGenerator()
        self.window_days = window_days
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.min_candles = min_candles_for_signal

    def run(self, df: pd.DataFrame) -> List[WindowResult]:
        """
        Run walk-forward backtest on the full dataset.

        Args:
            df: Full historical OHLCV DataFrame with DatetimeIndex.

        Returns:
            List of WindowResult, one per time window.
        """
        windows = self._split_windows(df)
        results = []

        for i, (start, end) in enumerate(windows):
            window_df = df[start:end].copy()
            logging.info(f"Window {i+1}/{len(windows)}: {start} → {end} ({len(window_df)} candles)")

            result = self._backtest_window(window_df, start, end)
            results.append(result)

            logging.info(
                f"  Trades: {result.n_trades} | Win rate: {result.win_rate:.1%} | "
                f"PNL: {result.total_pnl_pct:+.2%} | Max DD: {result.max_drawdown:.2%}"
            )

        self._print_summary(results)
        return results

    def _split_windows(self, df: pd.DataFrame):
        """Split full date range into non-overlapping windows."""
        start = df.index[0]
        end = df.index[-1]
        windows = []
        current = start
        while current < end:
            window_end = current + pd.Timedelta(days=self.window_days)
            if window_end > end:
                window_end = end
            windows.append((current.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')))
            current = window_end
        return windows

    def _backtest_window(self, df: pd.DataFrame, start: str, end: str) -> WindowResult:
        result = WindowResult(start=start, end=end)

        if len(df) < self.min_candles:
            logging.warning(f"Not enough candles ({len(df)}) for window {start}→{end}, skipping")
            return result

        # Enrich full window with regime indicators
        df = self.regime_detector.add_regime_to_df(df)

        in_position = False
        current_trade_signal: TradingSignal = None
        entry_time = None
        entry_price = None

        for i in range(self.min_candles, len(df)):
            candle = df.iloc[i]
            slice_df = df.iloc[:i + 1]  # Up to and including current candle

            if in_position:
                # Check TP/SL on current candle's high/low
                # Use high/low to simulate intra-candle touches
                hit_tp, hit_sl = self._check_exits(candle, current_trade_signal)

                if hit_tp or hit_sl:
                    exit_price = current_trade_signal.take_profit if hit_tp else current_trade_signal.stop_loss
                    exit_reason = 'tp' if hit_tp else 'sl'
                    trade = self._build_trade(
                        current_trade_signal, entry_time, candle.name,
                        entry_price, exit_price, exit_reason
                    )
                    result.trades.append(trade)
                    in_position = False
                    current_trade_signal = None

            else:
                # Generate signal for this candle
                regime = Regime(candle['regime'])
                signal = self.signal_gen.generate(slice_df, regime)

                if signal.signal != SignalType.HOLD:
                    in_position = True
                    current_trade_signal = signal
                    entry_time = candle.name
                    entry_price = float(candle['close'])

        # If still in position at end of window, close at last price
        if in_position and current_trade_signal is not None:
            last = df.iloc[-1]
            exit_price = float(last['close'])
            trade = self._build_trade(
                current_trade_signal, entry_time, last.name,
                entry_price, exit_price, 'end_of_data'
            )
            result.trades.append(trade)

        return result

    def _check_exits(self, candle, signal: TradingSignal):
        """
        Check if candle's high/low touched TP or SL.
        In a real exchange, either can be hit intra-candle.
        We conservatively give priority to SL (worst case).
        """
        high = float(candle['high'])
        low = float(candle['low'])
        is_long = signal.signal == SignalType.LONG

        if is_long:
            hit_sl = low <= signal.stop_loss
            hit_tp = high >= signal.take_profit
        else:
            hit_sl = high >= signal.stop_loss
            hit_tp = low <= signal.take_profit

        # If both hit same candle, assume SL (conservative)
        if hit_sl and hit_tp:
            return False, True  # SL wins

        return hit_tp, hit_sl

    def _build_trade(self, signal, entry_time, exit_time, entry_price, exit_price, exit_reason) -> Trade:
        is_long = signal.signal == SignalType.LONG
        direction = 'long' if is_long else 'short'

        # Raw price PNL %
        if is_long:
            raw_pnl = (exit_price - entry_price) / entry_price
        else:
            raw_pnl = (entry_price - exit_price) / entry_price

        # Fees: 2 sides (open + close)
        total_fee = 2 * self.fee_rate
        net_pnl = raw_pnl - total_fee
        net_pnl_leveraged = net_pnl * self.leverage

        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            atr=signal.atr,
            regime=signal.regime.value,
            exit_reason=exit_reason,
            pnl_pct=net_pnl,
            pnl_pct_leveraged=net_pnl_leveraged,
        )

    def _print_summary(self, results: List[WindowResult]):
        all_trades = [t for r in results for t in r.trades]
        if not all_trades:
            logging.warning("No trades generated across all windows")
            return

        total = len(all_trades)
        wins = sum(1 for t in all_trades if t.pnl_pct > 0)
        tp_exits = sum(1 for t in all_trades if t.exit_reason == 'tp')
        sl_exits = sum(1 for t in all_trades if t.exit_reason == 'sl')
        total_pnl = sum(t.pnl_pct_leveraged for t in all_trades)

        pnls = [t.pnl_pct_leveraged for t in all_trades]
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((peak - equity).max())

        print("\n" + "="*60)
        print("WALK-FORWARD BACKTEST SUMMARY")
        print("="*60)
        print(f"Windows tested:   {len(results)}")
        print(f"Total trades:     {total}")
        print(f"Win rate:         {wins/total:.1%}  ({wins}W / {total-wins}L)")
        print(f"TP exits:         {tp_exits} ({tp_exits/total:.1%})")
        print(f"SL exits:         {sl_exits} ({sl_exits/total:.1%})")
        print(f"Total PNL:        {total_pnl:+.2%} (leveraged, after fees)")
        print(f"Avg PNL/trade:    {total_pnl/total:+.2%}")
        print(f"Max drawdown:     {max_dd:.2%}")

        print(f"\nPer-regime breakdown:")
        regimes = pd.Series([t.regime for t in all_trades]).unique()
        for regime in sorted(regimes):
            r_trades = [t for t in all_trades if t.regime == regime]
            r_wins = sum(1 for t in r_trades if t.pnl_pct > 0)
            r_pnl = sum(t.pnl_pct_leveraged for t in r_trades)
            r_tp = sum(1 for t in r_trades if t.exit_reason == 'tp')
            r_sl = sum(1 for t in r_trades if t.exit_reason == 'sl')
            print(f"  {regime:<16} {len(r_trades):>3} trades | "
                  f"WR={r_wins/len(r_trades):.0%} | "
                  f"PNL={r_pnl:+.1%} | "
                  f"TP={r_tp} SL={r_sl}")

        print(f"\nPer-window PNL:")
        for r in results:
            marker = "OK" if r.total_pnl_pct >= 0 else "--"
            print(f"  [{marker}] {r.start} → {r.end}  "
                  f"{r.n_trades:>2} trades | WR={r.win_rate:.0%} | PNL={r.total_pnl_pct:+.1%}")
        print("="*60)

    def to_dataframe(self, results: List[WindowResult]) -> pd.DataFrame:
        """Export all trades to a DataFrame for analysis."""
        all_trades = [t for r in results for t in r.trades]
        if not all_trades:
            return pd.DataFrame()

        rows = []
        for t in all_trades:
            rows.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'take_profit': t.take_profit,
                'stop_loss': t.stop_loss,
                'atr': t.atr,
                'regime': t.regime,
                'exit_reason': t.exit_reason,
                'pnl_pct': t.pnl_pct,
                'pnl_pct_leveraged': t.pnl_pct_leveraged,
            })
        return pd.DataFrame(rows)
