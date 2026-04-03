"""
Grid Bot Live Manager

Manages a live grid on Binance Futures using limit orders.

Lifecycle:
  1. setup()   — calculate grid levels, place all BUY limit orders below price
  2. run()     — polling loop: detect fills, place new orders, log status
  3. teardown() — cancel all open orders, close any open positions

Order tracking:
  We maintain a dict {order_id → slot_index} for placed orders.
  Each loop we compare this against Binance's open orders list.
  Orders no longer open = filled → handle accordingly.
"""

import logging
import time
from typing import Dict, Optional

from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC

from crypto_bot.grid.grid_config import GridConfig
from crypto_bot.orders.orders_manager import OrdersManager


class GridManager:
    def __init__(self, binance_client: Client, config: GridConfig):
        self.client = binance_client
        self.config = config
        self.orders = OrdersManager(binance_client)

        self.levels = config.levels

        # {order_id: slot_index} — tracks all our active orders
        self.buy_orders: Dict[int, int] = {}   # order_id → slot
        self.sell_orders: Dict[int, int] = {}  # order_id → slot

        # P&L tracking
        self.completed_cycles = 0
        self.total_pnl_usdt = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self):
        """Place initial BUY limit orders at all grid levels below current price."""
        current_price = self.orders.get_current_price(self.config.symbol)
        logging.info(f"Setting up grid. Current price: ${current_price:.2f}")
        logging.info(self.config.summary())

        self.orders.futures_change_leverage(self.config.symbol, self.config.leverage)
        self._cancel_all_orders()

        for i, level in enumerate(self.levels[:-1]):  # all but the top level
            if level < current_price:
                self._place_buy(i)

        placed = len(self.buy_orders)
        logging.info(f"Grid setup complete: {placed} BUY orders placed")

    def run(self, poll_interval_secs: int = 30, max_iterations: int = -1):
        """Main polling loop."""
        iteration = 0
        consecutive_errors = 0

        while max_iterations == -1 or iteration < max_iterations:
            try:
                self._cycle()
                consecutive_errors = 0
                iteration += 1
                logging.info(
                    f"[{iteration}] Cycles: {self.completed_cycles} | "
                    f"PNL: ${self.total_pnl_usdt:+.2f} USDT"
                )
                time.sleep(poll_interval_secs)

            except Exception as e:
                consecutive_errors += 1
                logging.exception(f"Error in grid cycle ({consecutive_errors}): {e}")
                if consecutive_errors >= 5:
                    logging.critical("Too many errors — stopping grid")
                    break
                time.sleep(10)

        self.teardown()

    def teardown(self):
        """Cancel all open orders and close any residual positions."""
        logging.info("Tearing down grid...")
        self._cancel_all_orders()
        try:
            position = self.orders.get_current_position_info(self.config.symbol)
            if position and position.positionAmt != 0:
                self.orders.close_current_positions(self.config.symbol)
                logging.info("Closed residual position")
        except Exception as e:
            logging.error(f"Could not close position during teardown: {e}")
        logging.info(
            f"Grid stopped. Total cycles: {self.completed_cycles} | "
            f"Total PNL: ${self.total_pnl_usdt:+.2f} USDT"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cycle(self):
        """Detect fills and place new orders."""
        open_order_ids = self._get_open_order_ids()

        # Check BUY orders for fills
        filled_buy_slots = []
        for order_id, slot in list(self.buy_orders.items()):
            if order_id not in open_order_ids:
                logging.info(f"BUY filled at slot {slot} (${self.levels[slot]:.2f})")
                filled_buy_slots.append(slot)
                del self.buy_orders[order_id]

        # For each filled BUY: place SELL at next level up
        for slot in filled_buy_slots:
            sell_level = self.levels[slot + 1]
            self._place_sell(slot, sell_level)

        # Check SELL orders for fills
        filled_sell_slots = []
        for order_id, slot in list(self.sell_orders.items()):
            if order_id not in open_order_ids:
                buy_level = self.levels[slot]
                sell_level = self.levels[slot + 1]
                pnl = (
                    (sell_level - buy_level) / buy_level - 2 * self.config.fee_rate
                ) * self.config.usdt_per_grid

                logging.info(
                    f"SELL filled at slot {slot} (${sell_level:.2f}) | "
                    f"Cycle profit: ${pnl:+.3f} USDT"
                )
                self.completed_cycles += 1
                self.total_pnl_usdt += pnl
                filled_sell_slots.append(slot)
                del self.sell_orders[order_id]

        # For each filled SELL: place new BUY at same slot
        for slot in filled_sell_slots:
            self._place_buy(slot)

    def _place_buy(self, slot: int):
        price = self.levels[slot]
        qty = self.config.qty_at_level(slot)
        try:
            order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=qty,
                price=price,
                positionSide="BOTH",
            )
            self.buy_orders[order["orderId"]] = slot
            logging.info(f"BUY placed: slot={slot} price=${price:.2f} qty={qty}")
        except Exception as e:
            logging.error(f"Failed to place BUY at slot {slot} (${price:.2f}): {e}")

    def _place_sell(self, slot: int, price: float):
        qty = self.config.qty_at_level(slot)
        try:
            order = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=qty,
                price=price,
                positionSide="BOTH",
                reduceOnly=True,
            )
            self.sell_orders[order["orderId"]] = slot
            logging.info(f"SELL placed: slot={slot} price=${price:.2f} qty={qty}")
        except Exception as e:
            logging.error(f"Failed to place SELL at slot {slot} (${price:.2f}): {e}")

    def _get_open_order_ids(self) -> set:
        try:
            open_orders = self.client.futures_get_open_orders(symbol=self.config.symbol)
            return {o["orderId"] for o in open_orders}
        except Exception as e:
            logging.error(f"Could not fetch open orders: {e}")
            return set(self.buy_orders.keys()) | set(self.sell_orders.keys())

    def _cancel_all_orders(self):
        try:
            self.client.futures_cancel_all_open_orders(symbol=self.config.symbol)
            self.buy_orders.clear()
            self.sell_orders.clear()
            logging.info("All open orders cancelled")
        except Exception as e:
            logging.error(f"Could not cancel orders: {e}")
