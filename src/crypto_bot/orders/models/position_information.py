import logging

from pydantic import BaseModel, validator

# Binance Futures fees (without BNB discount)
TAKER_FEE_RATE = 0.0004  # 0.04% for market orders
MAKER_FEE_RATE = 0.0002  # 0.02% for limit orders
DEFAULT_FEE_RATE = TAKER_FEE_RATE


class PositionInformation(BaseModel):
    positionAmt: float
    unRealizedProfit: float
    entryPrice: float
    markPrice: float
    leverage: int = 0
    notional: float

    @validator('positionAmt')
    def position_amt_check(cls, v):
        return round(v, 3)

    @validator('unRealizedProfit')
    def unrealized_profit_check(cls, v):
        return round(v, 3)

    @validator('entryPrice')
    def entry_price_check(cls, v):
        return round(v, 3)

    @validator('markPrice')
    def mark_price_check(cls, v):
        return round(v, 3)

    @validator('notional')
    def notional_check(cls, v):
        return round(v, 3)


class TraderPosition:
    def __init__(self, qty, leverage, initial_price):
        self.qty = qty
        self.leverage = leverage
        self.initial_price = initial_price

    def get_position(self, current_price) -> PositionInformation:
        return PositionInformation(
            positionAmt=self.qty,
            unRealizedProfit=((current_price - self.initial_price) * self.qty),
            entryPrice=self.initial_price,
            markPrice=current_price,
            leverage=self.leverage,
            notional=current_price * self.qty,
        )

    @staticmethod
    def calculate_unrealized_pnl(position_info, include_fees=True, fee_rate=None):
        if position_info.notional == 0 or position_info.positionAmt == 0:
            return 0.0

        leverage = position_info.leverage if position_info.leverage > 0 else 1
        base_pnl = position_info.unRealizedProfit / (abs(position_info.notional) / leverage)

        if not include_fees:
            return round(base_pnl, 4)

        if fee_rate is None:
            fee_rate = DEFAULT_FEE_RATE

        total_fee_pct = 2 * fee_rate
        pnl_after_fees = round(base_pnl - total_fee_pct, 4)

        logging.info(
            f"Raw PNL: {base_pnl*100:.2f}%, Fees: -{total_fee_pct*100:.2f}%, Net: {pnl_after_fees*100:.2f}%"
        )
        return pnl_after_fees
