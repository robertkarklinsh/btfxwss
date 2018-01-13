import logging
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from btfxwss.queue_processor import QueueProcessor
from btfxwss.client import BtfxWss
from sqlalchemy import create_engine
from btfxwss.data.candle_writer import CandleWriter
from btfxwss.data.trade_writer import TradeWriter

logging.basicConfig(filename='test.log', level=logging.ERROR)
log = logging.getLogger(__name__)


class Sniffer(QueueProcessor):
    def __init__(self, data_q=None):
        super().__init__(data_q)
        self.book = defaultdict(pd.DataFrame)
        self.candle_handlers = []
        self.book_handlers = []
        self.trade_handlers = []
        self.timestamp_index = defaultdict(int)
        self.payload = defaultdict(self._array_factory)

    def _handle_candles(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        if payload.size == 0:
            return
        # check that data is not historical and then write candle when newer one arrives
        if len(payload.shape) != 2:
            if self.timestamp_index[symbol] == 0:
                self.timestamp_index[symbol] = payload[0]
                self.payload[symbol] = payload
            if self.timestamp_index[symbol] == payload[0]:
                self.payload[symbol] = payload
            if self.timestamp_index[symbol] < payload[0]:
                self.timestamp_index[symbol] = payload[0]
                self.payload[symbol] = self.payload[symbol].reshape((1, -1))
                # formatting payload to [[ts, o, h, l, c, v]]
                self.payload[symbol] = self.payload[symbol][:, [0, 1, 3, 4, 2, 5]]
                for handler in self.candle_handlers:
                    handler(symbol, self.payload[symbol])
                self.payload[symbol] = payload

                # TODO Fix missing candles with DateTime index
                # TODO handle historical candles at the beginning of connection

    def _handle_raw_book(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        if payload.size == 0:
            return
        # check whether received initial data or single entity
        if len(payload.shape) == 2:
            num_orders = payload.shape[0]
            index = payload.flatten('F')[np.arange(num_orders)]
            self.book[symbol] = pd.DataFrame(payload[:, 1:], index=index)
        if len(payload.shape) == 1:
            # if price is zero we have to delete this order from book
            if payload[1] == 0:
                self.book[symbol].drop(payload[0])
            else:
                self.book[symbol].loc[payload[0]] = payload[1:]
        for handler in self.book_handlers:
            handler(symbol, self.book[symbol])

    def _handle_trades(self, dtype, data, ts):
        self.log.debug("_handle_trades: %s - %s - %s", dtype, data, ts)
        channel_id, *payload = data
        # every trade is resent with trade update flag
        if payload[0] == 'tu':
            return
        else:
            payload = payload[-1]
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        # check if received initial data
        if payload.size == 0:
            return
        if len(payload.shape) == 2:
            for handler in self.trade_handlers:
                handler.write_initial_data(symbol, payload)
        if len(payload.shape) == 1:
            for handler in self.trade_handlers:
                handler.write_entity(symbol, payload)

    def _array_factory(self):
        return np.zeros(6)


if __name__ == "__main__":
    # engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
    engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo=True)

    candle_writer = CandleWriter(engine)
    trade_writer = TradeWriter(engine)
    time.sleep(1)
    sniffer = Sniffer()
    sniffer.candle_handlers.append(candle_writer.write)
    sniffer.trade_handlers.append(trade_writer)

    client = BtfxWss(queue_processor=sniffer, log_level=logging.ERROR)
    client.start()
    time.sleep(1)
    client.subscribe_to_candles('BTCUSD')
    client.subscribe_to_candles('ETHUSD')
    client.subscribe_to_candles('OMGUSD')
    client.subscribe_to_candles('IOTUSD')
    client.subscribe_to_candles('LTCUSD')
    client.subscribe_to_candles('XMRUSD')
    client.subscribe_to_candles('EDOUSD')
    client.subscribe_to_candles('AVTUSD')
    client.subscribe_to_candles('ZECUSD')
    client.subscribe_to_candles('XRPUSD')

    client.subscribe_to_trades('BTCUSD')
    client.subscribe_to_trades('ETHUSD')
    client.subscribe_to_trades('OMGUSD')
    client.subscribe_to_trades('IOTUSD')
    client.subscribe_to_trades('LTCUSD')
    client.subscribe_to_trades('XMRUSD')
    client.subscribe_to_trades('EDOUSD')
    client.subscribe_to_trades('AVTUSD')
    client.subscribe_to_trades('ZECUSD')
    client.subscribe_to_trades('XRPUSD')

    try:
        while True:
            time.sleep(10)
            candle_writer.commit()
            trade_writer.commit()
    except KeyboardInterrupt as e:
        logging.debug(str(e) + '\n' + 'Exiting...')
        client.stop()
