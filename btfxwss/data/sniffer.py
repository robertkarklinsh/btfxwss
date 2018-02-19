import logging
import time
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict
from btfxwss.utils.utils import btfx_ts_to_datetime
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
        self.first_dt_to_write = defaultdict(lambda: None)
        self.write_time_delta = datetime.timedelta(minutes=2)  # Must be even!
        self.global_init_stage = True
        self.init_stage = defaultdict(self._true_default_dict_factory)
        self.payload = None

    def _handle_candles(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        if payload.size == 0:
            return
        # check that data is not historical
        if len(payload.shape) != 2:
            data_dt = btfx_ts_to_datetime(payload[0])
            if self.global_init_stage:
                self.payload = defaultdict(self._df_factory(data_dt))
                self.global_init_stage = False
            if self.init_stage[symbol]:
                self.first_dt_to_write[symbol] = data_dt
                self.init_stage[symbol] = False
            else:
                self.payload[symbol].loc[data_dt] = payload[[1, 3, 4, 2, 5]]

            if data_dt > self.first_dt_to_write[symbol] + self.write_time_delta:
                last_dt_to_write = data_dt - self.write_time_delta / 2
                for handler in self.candle_handlers:
                    handler(symbol, self.payload[symbol].loc[self.first_dt_to_write[symbol]:last_dt_to_write])
                self.first_dt_to_write[symbol] = last_dt_to_write + datetime.timedelta(minutes=1)

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

    def _true_default_dict_factory(self):
        return True

    def _df_factory(self, start_dt):
        def callable():
            return pd.DataFrame(
                index=pd.date_range(start_dt - datetime.timedelta(minutes=5), start_dt + datetime.timedelta(360),
                                    freq='Min'),
                columns=['open', 'high', 'low', 'close', 'volume'],
                dtype=np.float32)

        return callable


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
    #
    # client.subscribe_to_trades('BTCUSD')
    # client.subscribe_to_trades('ETHUSD')
    # client.subscribe_to_trades('OMGUSD')
    # client.subscribe_to_trades('IOTUSD')
    # client.subscribe_to_trades('LTCUSD')
    # client.subscribe_to_trades('XMRUSD')
    # client.subscribe_to_trades('EDOUSD')
    # client.subscribe_to_trades('AVTUSD')
    # client.subscribe_to_trades('ZECUSD')
    # client.subscribe_to_trades('XRPUSD')

    try:
        while True:
            time.sleep(30)
            candle_writer.commit()
 #           trade_writer.commit()
    except KeyboardInterrupt as e:
        logging.debug(str(e) + '\n' + 'Exiting...')
        client.stop()
