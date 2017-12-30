import logging
from unittest import TestCase
import time
from queue import Empty
from btfxwss import BtfxWss
from websocket import WebSocketConnectionClosedException

logging.basicConfig(filename='test.log', level=logging.DEBUG)
log = logging.getLogger(__name__)


def populate_db(time_interval=30):
    wss = BtfxWss(log_level=logging.DEBUG)
    wss.start()
    time.sleep(1)
    # wss.subscribe_to_ticker('BTCUSD')
    wss.subscribe_to_candles('BTCUSD')
    # wss.subscribe_to_order_book('BTCUSD')
    # wss.subscribe_to_raw_order_book('BTCUSD')
    # wss.subscribe_to_trades('BTCUSD')
    time.sleep(10)

    start_time = time.time()
    while time.time() < start_time + time_interval:
        queue = wss.candles('BTCUSD')
        data, ts = queue.get()
        log.debug(data)

if __name__ == '__main__':
    populate_db(10)

