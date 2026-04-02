from crypto_bot.orders.models.position_information import PositionInformation


class TradingStatus:
    def __init__(self):
        self.closed_positions = []
        self.current_position: PositionInformation = None
        self.accumulated_profit = 0.0

    def open_position(self, position_info: PositionInformation):
        self.current_position = position_info

    def close_position(self, position_info: PositionInformation):
        self.current_position = None
        self.closed_positions.append(position_info)
        self.accumulated_profit += position_info.unRealizedProfit

    def get_balance(self) -> float:
        if self.current_position:
            return self.accumulated_profit + self.current_position.unRealizedProfit
        return self.accumulated_profit

    def has_open_position(self) -> bool:
        return self.current_position is not None

    def status_update(self, current_price: float):
        if self.current_position:
            self.current_position.unRealizedProfit = round(
                (current_price - self.current_position.entryPrice) * self.current_position.positionAmt, 3
            )
            self.current_position.markPrice = current_price
            self.current_position.notional = round(current_price * self.current_position.positionAmt, 3)
