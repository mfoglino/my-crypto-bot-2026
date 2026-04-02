import logging
import time
from functools import wraps

from binance.enums import *
from requests.exceptions import ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

from crypto_bot.orders.models.position_information import PositionInformation

RECVWINDOW = 60000


def retry_on_connection_error(max_retries=3, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, Timeout, ProtocolError, Exception) as e:
                    last_exception = e
                    error_msg = str(e)

                    is_connection_error = (
                        isinstance(e, (ConnectionError, Timeout, ProtocolError))
                        or 'Connection aborted' in error_msg
                        or 'Remote end closed connection' in error_msg
                        or 'RemoteDisconnected' in error_msg
                    )

                    if not is_connection_error:
                        raise

                    if attempt < max_retries:
                        logging.warning(f"Connection error in {func.__name__} (attempt {attempt+1}/{max_retries+1}): {error_msg}")
                        logging.info(f"Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"All {max_retries+1} attempts failed for {func.__name__}: {error_msg}")
                        raise last_exception

            raise last_exception
        return wrapper
    return decorator


class OrdersManager:
    def __init__(self, binance_client):
        self.client = binance_client

    @retry_on_connection_error()
    def create_long_order(self, symbol, qty):
        order = self.client.futures_create_order(
            symbol=symbol, side=SIDE_BUY, positionSide='BOTH',
            type=ORDER_TYPE_MARKET, quantity=qty
        )
        logging.info(f"LONG market order: orderId={order['orderId']}, qty={order['origQty']}, filled={order['executedQty']}")
        self._verify_fill(symbol, order)
        return order

    @retry_on_connection_error()
    def create_short_order(self, symbol, qty):
        order = self.client.futures_create_order(
            symbol=symbol, side=SIDE_SELL, positionSide='BOTH',
            type=ORDER_TYPE_MARKET, quantity=qty
        )
        logging.info(f"SHORT market order: orderId={order['orderId']}, qty={order['origQty']}, filled={order['executedQty']}")
        self._verify_fill(symbol, order)
        return order

    def _verify_fill(self, symbol, order):
        if float(order.get('executedQty', 0)) == 0:
            logging.warning(f"Order {order['orderId']} not immediately filled, rechecking...")
            time.sleep(2)
            try:
                status = self.client.futures_get_order(symbol=symbol, orderId=order['orderId'])
                if float(status.get('executedQty', 0)) > 0:
                    order['executedQty'] = status['executedQty']
                else:
                    logging.error(f"Order {order['orderId']} FAILED TO FILL. Status: {status['status']}")
            except Exception as e:
                logging.error(f"Error verifying fill: {e}")

    @retry_on_connection_error()
    def close_long_position(self, symbol, qty):
        order = self.client.futures_create_order(
            symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET,
            quantity=qty, positionSide='BOTH', reduceOnly='true'
        )
        logging.info(f"Closed LONG: orderId={order['orderId']}, qty={order['origQty']}")
        return order

    @retry_on_connection_error()
    def close_short_position(self, symbol, qty):
        order = self.client.futures_create_order(
            symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET,
            quantity=qty, positionSide='BOTH', reduceOnly='true'
        )
        logging.info(f"Closed SHORT: orderId={order['orderId']}, qty={order['origQty']}")
        return order

    @retry_on_connection_error()
    def close_current_positions(self, symbol):
        position_info = self.get_current_position_info(symbol)
        if position_info is None:
            return
        amt = position_info.positionAmt
        self.client.futures_cancel_all_open_orders(symbol=symbol)
        if amt > 0:
            self.close_long_position(symbol, abs(amt))
        elif amt < 0:
            self.close_short_position(symbol, abs(amt))

    @retry_on_connection_error()
    def get_current_position_info(self, symbol) -> PositionInformation | None:
        positions = self.client.futures_position_information(recvWindow=RECVWINDOW)
        filtered = [p for p in positions if p['symbol'] == symbol]
        if not filtered:
            return None
        return PositionInformation(**filtered[0])

    @retry_on_connection_error()
    def is_there_an_open_position(self, symbol) -> bool:
        info = self.get_current_position_info(symbol)
        return info is not None and info.positionAmt != 0

    @retry_on_connection_error()
    def get_current_price(self, symbol) -> float:
        ticker = self.client.futures_ticker(symbol=symbol)
        return float(ticker['lastPrice'])

    @retry_on_connection_error()
    def futures_change_leverage(self, symbol, leverage):
        return self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    @retry_on_connection_error()
    def get_usdt_balance(self) -> float:
        balances = self.client.futures_account_balance(recvWindow=8000)
        usdt = next((b for b in balances if b['asset'] == 'USDT'), None)
        return float(usdt['balance']) if usdt else 0.0

    @staticmethod
    def qty_from_usdt(price, usdt_amount) -> float:
        return round(usdt_amount / price, 3)
