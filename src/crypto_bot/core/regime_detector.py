"""
Market Regime Detector

Classifies the market into one of three states:
  - TRENDING_UP:   ADX > threshold AND EMA slope positive
  - TRENDING_DOWN: ADX > threshold AND EMA slope negative
  - RANGING:       ADX < threshold (sideways/choppy market)

This is the core innovation vs the 2025 bot. Instead of applying the same
signal logic in all conditions, we first determine *what kind of market* we're
in and then pick the right strategy for that regime.
"""

from enum import Enum
import pandas as pd
import numpy as np
import logging


class Regime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    NEUTRAL = "neutral"  # ADX in gray zone — skip trading


class RegimeDetector:
    """
    Detects market regime using ADX + EMA slope.

    Parameters
    ----------
    adx_period : int
        Lookback period for ADX (standard is 14)
    adx_trend_threshold : float
        ADX above this = trending market (standard is 25)
    adx_range_threshold : float
        ADX below this = ranging market (standard is 20)
        Values between range and trend thresholds = neutral (skip)
    ema_slope_period : int
        EMA period used for slope calculation
    ema_slope_lookback : int
        Number of candles to measure slope over
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        ema_slope_period: int = 50,
        ema_slope_lookback: int = 5,
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.ema_slope_period = ema_slope_period
        self.ema_slope_lookback = ema_slope_lookback

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ADX (Average Directional Index).
        ADX measures trend *strength*, not direction.
        High ADX = strong trend (up or down). Low ADX = ranging.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()

        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        # Smoothed (Wilder's smoothing = EWM with alpha=1/period)
        period = self.adx_period
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr
        di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr

        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).fillna(0)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx

    def calculate_ema_slope(self, df: pd.DataFrame) -> pd.Series:
        """
        EMA slope: positive = uptrend, negative = downtrend.
        Normalized by price so it's comparable across different price levels.
        """
        ema = df['close'].ewm(span=self.ema_slope_period, adjust=False).mean()
        slope = (ema - ema.shift(self.ema_slope_lookback)) / ema.shift(self.ema_slope_lookback)
        return slope

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR for dynamic stop/take-profit sizing."""
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the dataframe with regime columns:
          - adx
          - ema_slope
          - atr
          - regime (Regime enum value as string)
        """
        df = df.copy()
        df['adx'] = self.calculate_adx(df)
        df['ema_slope'] = self.calculate_ema_slope(df)
        df['atr'] = self.calculate_atr(df)

        def classify(row):
            if pd.isna(row['adx']) or pd.isna(row['ema_slope']):
                return Regime.NEUTRAL.value
            if row['adx'] >= self.adx_trend_threshold:
                return Regime.TRENDING_UP.value if row['ema_slope'] > 0 else Regime.TRENDING_DOWN.value
            elif row['adx'] < self.adx_range_threshold:
                return Regime.RANGING.value
            else:
                return Regime.NEUTRAL.value  # gray zone: 20 < ADX < 25

        df['regime'] = df.apply(classify, axis=1)

        # Log distribution for debugging
        counts = df['regime'].value_counts()
        logging.info(f"Regime distribution: {counts.to_dict()}")

        return df

    def get_current_regime(self, df: pd.DataFrame) -> Regime:
        """Returns the regime for the most recent candle."""
        enriched = self.add_regime_to_df(df)
        last = enriched.iloc[-1]
        regime = Regime(last['regime'])
        logging.info(
            f"Current regime: {regime.value} | ADX={last['adx']:.1f} | "
            f"EMA slope={last['ema_slope']*100:.3f}% | ATR={last['atr']:.2f}"
        )
        return regime
