"""
Regime-Aware Signal Generator

Generates trading signals based on the current market regime:

  TRENDING_UP:   LONG on pullback to EMA21 (price dips to EMA21, RSI 40-60)
  TRENDING_DOWN: SHORT on bounce to EMA21 (price bounces to EMA21, RSI 40-60)
  RANGING:       RSI extremes + Bollinger Band touches

Why pullback instead of crossover?
  Crossover entry = entering at the peak of the initial move → high chance of
  immediate reversal / SL hit (confirmed by 27% win rate in v1).
  Pullback entry = price retraces to the moving average, trend resumes →
  better price, more room before SL, trend already confirmed.
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
        Stop loss distance = atr * this. Default 1.0
    atr_tp_multiplier : float
        Take profit distance = atr * this. Default 2.0 → R:R = 2:1
    min_rr : float
        Minimum acceptable R:R. Signals below this are filtered.
    rsi_oversold : float
        RSI threshold for RANGING LONG. Default 30.
    rsi_overbought : float
        RSI threshold for RANGING SHORT. Default 70.
    ema_fast : int
        Fast EMA period. Default 9.
    ema_slow : int
        Slow EMA period used as pullback target. Default 21.
    pullback_tolerance : float
        How close to EMA21 price must be to count as a pullback.
        0.005 = within 0.5% of EMA21. Default 0.005.
    rsi_pullback_min : float
        RSI floor for trend pullback entries. Below this = possible reversal.
    rsi_pullback_max : float
        RSI ceiling for trend pullback entries. Above this = still extended.
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
        pullback_tolerance: float = 0.005,
        rsi_pullback_min: float = 40.0,
        rsi_pullback_max: float = 60.0,
    ):
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.min_rr = min_rr
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.pullback_tolerance = pullback_tolerance
        self.rsi_pullback_min = rsi_pullback_min
        self.rsi_pullback_max = rsi_pullback_max

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f'ema_{self.ema_fast}'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df[f'ema_{self.ema_slow}'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        # RSI (Wilder's smoothing via ewm com=period-1)
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

        return df

    def generate(self, df: pd.DataFrame, regime: Regime) -> TradingSignal:
        df = self._add_indicators(df)
        last = df.iloc[-1]

        current_price = float(last['close'])
        atr = float(last['atr'])

        if regime == Regime.NEUTRAL:
            return self._hold(current_price, atr, regime, "Regime NEUTRAL — skipping")

        if regime == Regime.TRENDING_DOWN:
            return self._trend_signal(last, current_price, atr, regime)
        else:
            # TRENDING_UP disabled: consistently 26% WR across all tests, -12.6% PNL
            # RANGING disabled: 27% WR, losing after fees
            return self._hold(current_price, atr, regime, f"{regime.value} — skipping (below break-even historically)")

    def _trend_signal(self, last, price, atr, regime) -> TradingSignal:
        """
        Pullback-to-EMA21 entry:

        TRENDING_UP:
          - EMA9 > EMA21 (uptrend confirmed by alignment)
          - Price pulled back to within `pullback_tolerance` of EMA21
          - RSI in [40, 60]: cooled off but not breaking down
          → LONG from the pullback

        TRENDING_DOWN:
          - EMA9 < EMA21 (downtrend confirmed)
          - Price bounced up to within `pullback_tolerance` of EMA21
          - RSI in [40, 60]: bounced but not overbought
          → SHORT from the bounce
        """
        ema_fast = float(last[f'ema_{self.ema_fast}'])
        ema_slow = float(last[f'ema_{self.ema_slow}'])
        rsi = float(last['rsi'])

        candle_low = float(last['low'])
        candle_high = float(last['high'])

        if regime == Regime.TRENDING_UP:
            trend_aligned = ema_fast > ema_slow
            if not trend_aligned:
                return self._hold(price, atr, regime, f"EMA alignment broken (EMA{self.ema_fast} < EMA{self.ema_slow})")

            # Require candle to have touched EMA21 (low <= EMA21) and closed back above it.
            # This confirms support held — not just "being near" EMA21.
            touched_support = candle_low <= ema_slow * (1 + self.pullback_tolerance)
            bounced = price >= ema_slow  # close is back above EMA21
            rsi_ok = rsi < self.rsi_pullback_max  # not overbought

            if touched_support and bounced and rsi_ok:
                sl = price - atr * self.atr_sl_multiplier
                tp = price + atr * self.atr_tp_multiplier
                signal = TradingSignal(
                    signal=SignalType.LONG, regime=regime,
                    entry_price=price, take_profit=tp, stop_loss=sl,
                    atr=atr,
                    reason=f"Candle touched EMA{self.ema_slow} and bounced (RSI={rsi:.1f})"
                )
                return self._filter_rr(signal, price, atr, regime)

            return self._hold(price, atr, regime,
                f"Uptrend — waiting for EMA{self.ema_slow} touch+bounce "
                f"(low={candle_low:.2f} EMA={ema_slow:.2f} close={price:.2f})")

        else:  # TRENDING_DOWN
            trend_aligned = ema_fast < ema_slow
            if not trend_aligned:
                return self._hold(price, atr, regime, f"EMA alignment broken (EMA{self.ema_fast} > EMA{self.ema_slow})")

            # Require candle to have touched EMA21 (high >= EMA21) and closed back below it.
            # Confirms resistance held.
            touched_resistance = candle_high >= ema_slow * (1 - self.pullback_tolerance)
            rejected = price <= ema_slow  # close is back below EMA21
            rsi_ok = rsi > self.rsi_pullback_min  # not oversold

            if touched_resistance and rejected and rsi_ok:
                sl = price + atr * self.atr_sl_multiplier
                tp = price - atr * self.atr_tp_multiplier
                signal = TradingSignal(
                    signal=SignalType.SHORT, regime=regime,
                    entry_price=price, take_profit=tp, stop_loss=sl,
                    atr=atr,
                    reason=f"Candle touched EMA{self.ema_slow} and was rejected (RSI={rsi:.1f})"
                )
                return self._filter_rr(signal, price, atr, regime)

            return self._hold(price, atr, regime,
                f"Downtrend — waiting for EMA{self.ema_slow} touch+rejection "
                f"(high={candle_high:.2f} EMA={ema_slow:.2f} close={price:.2f})")

    def _range_signal(self, last, price, atr, regime) -> TradingSignal:
        """
        Mean-reversion in ranging markets:
        - LONG: RSI < oversold AND price near BB lower band
        - SHORT: RSI > overbought AND price near BB upper band
        """
        rsi = float(last['rsi'])
        bb_upper = float(last['bb_upper'])
        bb_lower = float(last['bb_lower'])

        near_lower = price <= bb_lower * 1.002
        near_upper = price >= bb_upper * 0.998

        if rsi < self.rsi_oversold and near_lower:
            sl = price - atr * self.atr_sl_multiplier
            tp = price + atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.LONG, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"RSI oversold ({rsi:.1f}) + BB lower touch"
            )
            return self._filter_rr(signal, price, atr, regime)

        if rsi > self.rsi_overbought and near_upper:
            sl = price + atr * self.atr_sl_multiplier
            tp = price - atr * self.atr_tp_multiplier
            signal = TradingSignal(
                signal=SignalType.SHORT, regime=regime,
                entry_price=price, take_profit=tp, stop_loss=sl,
                atr=atr, reason=f"RSI overbought ({rsi:.1f}) + BB upper touch"
            )
            return self._filter_rr(signal, price, atr, regime)

        return self._hold(price, atr, regime, f"Ranging — RSI={rsi:.1f}, no extreme reached")

    def _filter_rr(self, signal: TradingSignal, price, atr, regime) -> TradingSignal:
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
