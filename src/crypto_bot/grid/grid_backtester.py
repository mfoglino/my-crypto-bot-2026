"""
Grid Bot Backtester

Simulates the grid strategy on historical OHLCV data.

Logic per candle:
  For each slot i (BUY at levels[i], SELL at levels[i+1]):
    - If NOT in position and candle.low <= levels[i]  → BUY fills
    - If IN position     and candle.high >= levels[i+1] → SELL fills → profit

Conservative assumptions:
  - When both BUY and SELL trigger in the same candle: BUY fills first,
    then SELL fills in the same candle (price oscillated). Counts as a cycle.
  - Entry price = grid level (limit order, not market)
  - No slippage beyond the limit price
  - Fee charged as taker on both sides (conservative; limit orders = maker)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from crypto_bot.grid.grid_config import GridConfig


@dataclass
class GridTrade:
    slot: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl_usdt: float
    pnl_pct: float           # Net % on invested capital (unlevered)
    pnl_pct_leveraged: float # Net % with leverage applied


@dataclass
class GridBacktestResult:
    config: GridConfig
    trades: List[GridTrade] = field(default_factory=list)
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_pnl_usdt(self) -> float:
        return sum(t.pnl_usdt for t in self.trades)

    @property
    def total_pnl_pct(self) -> float:
        return sum(t.pnl_pct_leveraged for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.pnl_pct_leveraged for t in self.trades])
        peak = np.maximum.accumulate(equity)
        return float((peak - equity).max())

    @property
    def cycles_per_day(self) -> float:
        if not self.trades or self.start is None or self.end is None:
            return 0.0
        days = (self.end - self.start).days or 1
        return self.n_trades / days

    def summary(self) -> str:
        days = (self.end - self.start).days if self.start and self.end else 0
        lines = [
            "",
            "=" * 60,
            "GRID BOT BACKTEST RESULTS",
            "=" * 60,
            self.config.summary(),
            "",
            f"Period:           {self.start.date()} → {self.end.date()} ({days} days)",
            f"Completed cycles: {self.n_trades}",
            f"Cycles/day:       {self.cycles_per_day:.2f}",
            f"Total PNL:        ${self.total_pnl_usdt:+.2f} USDT  "
            f"({self.total_pnl_pct:+.2%} leveraged)",
            f"Avg PNL/cycle:    ${self.total_pnl_usdt/self.n_trades:+.3f}" if self.n_trades else "Avg PNL/cycle: n/a",
            f"Max drawdown:     {self.max_drawdown:.2%}",
            "=" * 60,
        ]

        if self.trades:
            monthly = {}
            for t in self.trades:
                key = t.exit_time.strftime("%Y-%m")
                if key not in monthly:
                    monthly[key] = {"cycles": 0, "pnl": 0.0}
                monthly[key]["cycles"] += 1
                monthly[key]["pnl"] += t.pnl_usdt

            lines.append("\nMonthly breakdown:")
            for ym in sorted(monthly):
                m = monthly[ym]
                marker = "OK" if m["pnl"] >= 0 else "--"
                lines.append(
                    f"  [{marker}] {ym}  {m['cycles']:>3} cycles | "
                    f"PNL=${m['pnl']:+.2f}"
                )

            lines.append("\nPer-slot cycles:")
            slot_counts = {}
            for t in self.trades:
                slot_counts[t.slot] = slot_counts.get(t.slot, 0) + 1
            levels = self.config.levels
            for i in range(self.config.n_grids):
                count = slot_counts.get(i, 0)
                lines.append(
                    f"  Slot {i}: ${levels[i]:,.0f}→${levels[i+1]:,.0f}  "
                    f"{count} cycles"
                )

        return "\n".join(lines)


class GridBacktester:
    def run(self, df: pd.DataFrame, config: GridConfig) -> GridBacktestResult:
        """
        Run grid simulation on historical OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume],
                DatetimeIndex, sorted ascending.
            config: GridConfig with range, n_grids, capital.

        Returns:
            GridBacktestResult with all completed cycles.
        """
        levels = config.levels
        result = GridBacktestResult(
            config=config,
            start=df.index[0],
            end=df.index[-1],
        )

        # State per slot: None = BUY order active, pd.Timestamp = in position (entry time)
        in_position = [False] * config.n_grids
        entry_times = [None] * config.n_grids

        starting_price = float(df.iloc[0]["close"])

        for timestamp, row in df.iterrows():
            low = float(row["low"])
            high = float(row["high"])

            for i in range(config.n_grids):
                buy_level = levels[i]
                sell_level = levels[i + 1]

                # Only activate slots below starting price
                # (we start with BUY orders below current price)
                if buy_level >= starting_price:
                    continue

                if not in_position[i]:
                    # Waiting to buy at buy_level
                    if low <= buy_level:
                        in_position[i] = True
                        entry_times[i] = timestamp

                        # Check if same candle also hits sell_level (full cycle in one candle)
                        if high >= sell_level:
                            trade = self._build_trade(
                                config, i, entry_times[i], timestamp,
                                buy_level, sell_level
                            )
                            result.trades.append(trade)
                            in_position[i] = False
                            entry_times[i] = None

                else:
                    # In position — waiting for sell at sell_level
                    if high >= sell_level:
                        trade = self._build_trade(
                            config, i, entry_times[i], timestamp,
                            buy_level, sell_level
                        )
                        result.trades.append(trade)
                        in_position[i] = False
                        entry_times[i] = None

        logging.info(
            f"Grid backtest complete: {result.n_trades} cycles, "
            f"PNL=${result.total_pnl_usdt:+.2f} over {(result.end - result.start).days} days"
        )
        return result

    def _build_trade(
        self, config: GridConfig, slot: int,
        entry_time, exit_time,
        entry_price: float, exit_price: float
    ) -> GridTrade:
        qty = config.qty_at_level(slot)
        gross_pct = (exit_price - entry_price) / entry_price
        fee_pct = 2 * config.fee_rate
        net_pct = gross_pct - fee_pct
        net_pct_leveraged = net_pct * config.leverage
        pnl_usdt = net_pct * config.usdt_per_grid

        return GridTrade(
            slot=slot,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            pnl_usdt=pnl_usdt,
            pnl_pct=net_pct,
            pnl_pct_leveraged=net_pct_leveraged,
        )
