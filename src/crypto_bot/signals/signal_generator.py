"""
Regime-Aware Signal Generator

Generates trading signals based on the current market regime:

  TRENDING_UP:   Look for LONG entries on pullbacks to EMA21
  TRENDING_DOWN: Look for SHORT entries on bounces to EMA21
  RANGING:       RSI extremes + Bollinger Band touches

For every signal, we also compute ATR-based TP and SL so the
TradingManager knows exactly where to exit.

Minimum R:R enforced: 2:1 (TP = 2 * SL distance).
Trades that don't meet this are filtered out.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import logging

from crypto_bot.core.regime_detector import Regime


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


@dataclass
class TradingSignal:
    signal: SignalType
    regime: Regime
    entry_price: float
    take_profit: float
    stop_loss: float
    atr: float
    reason: str

    @property
    def risk_reward(self) -> float:
        if self.signal == SignalType.LONG:
            reward = self.take_profit - self.entry_price
            risk = self.entry_price - self.stop_loss
        elif self.signal == SignalType.SHORT:
            reward = self.entry_price - self.take_profit
            risk = self.stop_loss - self.entry_price
        else:
            return 0.0
        return reward / risk if risk > 0 else 0.0


class SignalGenerator:
    """
    Parameters
    ----------
    atr_sl_multiplier : float
        Stop loss = entry ± (atr * this). Default 1.0
    atr_tp_multiplier : float
        Take profit = entry ± (atr * this). Default 2.0 → R:R = 2:1
    min_rr : float
        Minimum acceptable R:R ratio. Signals below this are filtered.
    rsi_oversold : float
        RSI threshold for RANGING LONG signals. Default 30.
    rsi_overbought : float
        RSI threshold for RANGING SHORT signals. Default 70.
    ema_fast : int
        Fast EMA for trend-following entries.
    ema_slow : int
        Slow EMA for trend-following entries (pullback target).
    """

    def __init__(
        self,
        atr_sl_multiplier: float = 1.0,
        atr_tp_multiplier: float = 2.0,
        min_rr: float = 1.8,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ):
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.min_rr = min_rr
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f'ema_{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema_{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(com=13, adjust=False).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20

        return df

    def generate(self, df: pd.DataFrame, regime: Regime) -> TradingSignal:
        """
        Generate a signal for the current candle given the detected regime.

        Args:
            df: OHLCV dataframe, enriched with 'atr' column from RegimeDetector.
            regime: Current market regime.

        Returns:
            TradingSignal with entry, TP, SL and reason.
        """
        df = self._add_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(last['close'])
        atr = float(last['atr'])

        if regime == Regime.NEUTRAL:
            return self._hold(current_price, atr, regime, "Regime is NEUTRAL — waiting for clearer trend")

        if regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            return self._trend_signal(last, prev, current_price, atr, regime)
        else:  # RANGING
            return self._range_signal(last, prev, current_price, atr, regime)

    def _trend_signal(self, last, prev, price, atr, regime) -> TradingSignal:
        """
        Trend-following logic:
        - LONG: EMA9 crosses above EMA21 (bullish crossover)
        - SHORT: EMA9 crosses below EMA21 (bearish crossover)

        We only take crossovers — not chasing existing trends.
        This naturally filters out overtrading in strong trends.
        """
        ema_fast_now = float(last[f'ema_{self.ema_fast}'])
        ema_slow_now = float(last[f'ema_{self.ema_slow}'])
        ema_fast_prev = float(prev[f'ema_{self.ema_fast}'])
        ema_slow_prev = float(prev[f'ema_{self.ema_slow}'])

        bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
        bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now

        if regime == Regime.TRENDING_UP and bullish_cross:
            sl = price - atr * self.atr_sl_multiplier
            tp = price + atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.LONG, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"EMA{self.ema_fast}/EMA{self.ema_slow} bullish crossover in uptrend"
            )
            return self._filter_rr(signal, price, atr, regime)

        if regime == Regime.TRENDING_DOWN and bearish_cross:
            sl = price + atr * self.atr_sl_multiplier
            tp = price - atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.SHORT, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"EMA{self.ema_fast}/EMA{self.ema_slow} bearish crossover in downtrend"
            )
            return self._filter_rr(signal, price, atr, regime)

        direction = "up" if regime == Regime.TRENDING_UP else "down"
        return self._hold(price, atr, regime, f"Trending {direction} — waiting for EMA crossover")

    def _range_signal(self, last, prev, price, atr, regime) -> TradingSignal:
        """
        Mean-reversion logic for ranging markets:
        - LONG: RSI < oversold AND price touches lower Bollinger Band
        - SHORT: RSI > overbought AND price touches upper Bollinger Band
        """
        rsi = float(last['rsi'])
        bb_upper = float(last['bb_upper'])
        bb_lower = float(last['bb_lower'])

        near_lower = price <= bb_lower * 1.002  # within 0.2% of lower band
        near_upper = price >= bb_upper * 0.998  # within 0.2% of upper band

        if rsi < self.rsi_oversold and near_lower:
            sl = price - atr * self.atr_sl_multiplier
            tp = price + atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.LONG, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"RSI oversold ({rsi:.1f}) + BB lower touch in ranging market"
            )
            return self._filter_rr(signal, price, atr, regime)

        if rsi > self.rsi_overbought and near_upper:
            sl = price + atr * self.atr_sl_multiplier
            tp = price - atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.SHORT, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"RSI overbought ({rsi:.1f}) + BB upper touch in ranging market"
            )
            return self._filter_rr(signal, price, atr, regime)

        return self._hold(price, atr, regime, f"Ranging — RSI={rsi:.1f}, no extreme reached")

    def _filter_rr(self, signal: TradingSignal, price, atr, regime) -> TradingSignal:
        """Reject the signal if R:R is below minimum."""
        rr = signal.risk_reward
        if rr < self.min_rr:
            logging.warning(f"Signal rejected: R:R={rr:.2f} < minimum {self.min_rr}")
            return self._hold(price, atr, regime, f"R:R too low ({rr:.2f})")
        logging.info(f"Signal accepted: {signal.signal.value} | R:R={rr:.2f} | {signal.reason}")
        return signal

    def _hold(self, price, atr, regime, reason) -> TradingSignal:
        return TradingSignal(
            signal=SignalType.HOLD, regime=regime,
            entry_price=price, take_profit=price, stop_loss=price,
            atr=atr, reason=reason
        )
