"""
Grid Bot Configuration

A grid bot divides a price range into N equidistant levels and:
  - Places a BUY limit order at each level below current price
  - When a BUY fills → places a SELL at the next level up (take profit)
  - When a SELL fills → places a new BUY at that level (repeat)

Profit per completed cycle = grid_spacing - fees
The bot earns on every price oscillation within the range.

Risk: if price falls below lower_bound, all capital is deployed in
losing long positions. Set lower_bound conservatively.
"""

from dataclasses import dataclass
from typing import List

from crypto_bot.orders.models.position_information import TAKER_FEE_RATE


@dataclass
class GridConfig:
    symbol: str
    price_lower: float       # Bottom of the grid range
    price_upper: float       # Top of the grid range
    n_grids: int             # Number of grid levels (= number of BUY orders)
    total_investment_usdt: float  # Total USDT capital to deploy across all grids
    leverage: int            # Futures leverage (keep low: 1-3x)
    fee_rate: float = TAKER_FEE_RATE  # Per-side fee (0.04% taker)

    def __post_init__(self):
        if self.price_lower >= self.price_upper:
            raise ValueError(f"price_lower ({self.price_lower}) must be < price_upper ({self.price_upper})")
        if self.n_grids < 2:
            raise ValueError("n_grids must be >= 2")
        if self.total_investment_usdt <= 0:
            raise ValueError("total_investment_usdt must be > 0")

    @property
    def grid_spacing(self) -> float:
        """Price distance between adjacent grid levels."""
        return (self.price_upper - self.price_lower) / self.n_grids

    @property
    def usdt_per_grid(self) -> float:
        """USDT allocated per grid slot."""
        return self.total_investment_usdt / self.n_grids

    @property
    def levels(self) -> List[float]:
        """All N+1 price levels from lower to upper."""
        return [
            round(self.price_lower + i * self.grid_spacing, 2)
            for i in range(self.n_grids + 1)
        ]

    def qty_at_level(self, level_index: int) -> float:
        """ETH quantity for grid slot i (buy at levels[i])."""
        buy_price = self.levels[level_index]
        notional = self.usdt_per_grid * self.leverage
        return round(notional / buy_price, 3)

    def gross_profit_per_cycle_pct(self) -> float:
        """Gross profit % per completed buy→sell cycle (before fees)."""
        mid = (self.price_lower + self.price_upper) / 2
        return self.grid_spacing / mid

    def net_profit_per_cycle_pct(self) -> float:
        """Net profit % per completed cycle, after open+close fees."""
        return self.gross_profit_per_cycle_pct() - 2 * self.fee_rate

    def summary(self) -> str:
        levels = self.levels
        mid = (self.price_lower + self.price_upper) / 2
        lines = [
            f"Grid Config — {self.symbol}",
            f"  Range:        ${self.price_lower:,.0f} → ${self.price_upper:,.0f}",
            f"  Grids:        {self.n_grids} levels, ${self.grid_spacing:.1f} apart",
            f"  Spacing:      {self.grid_spacing/mid*100:.2f}% per grid",
            f"  Capital:      ${self.total_investment_usdt:.0f} total / ${self.usdt_per_grid:.1f} per grid",
            f"  Leverage:     {self.leverage}x",
            f"  Profit/cycle: {self.net_profit_per_cycle_pct()*100:.3f}% net (after fees)",
            f"  Levels:       {[f'${l:,.0f}' for l in levels]}",
        ]
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, d: dict) -> "GridConfig":
        return cls(
            symbol=d["symbol"],
            price_lower=float(d["price_lower"]),
            price_upper=float(d["price_upper"]),
            n_grids=int(d["n_grids"]),
            total_investment_usdt=float(d["total_investment_usdt"]),
            leverage=int(d["leverage"]),
            fee_rate=float(d.get("fee_rate", TAKER_FEE_RATE)),
        )
