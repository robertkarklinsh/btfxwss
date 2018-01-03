import logging
import time
import numpy as np
from collections import defaultdict
from btfxwss.queue_processor import QueueProcessor
from btfxwss.client import BtfxWss
from sqlalchemy import create_engine
from btfxwss.data.candle_writer import CandleWriter

logging.basicConfig(filename='test.log', level=logging.ERROR)
log = logging.getLogger(__name__)


class Sniffer(QueueProcessor):
    def __init__(self, data_q=None):
        super().__init__(data_q)
        self.candle_handlers = []
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

        # TODO handle historical candles at the beginning of connection


    def _array_factory(self):
        return np.zeros(6)


if __name__ == "__main__":
    # engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
    engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo=True)

    candle_writer = CandleWriter(engine)
    time.sleep(1)
    sniffer = Sniffer()
    sniffer.candle_handlers.append(candle_writer.write)

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

    try:
        while True:
            time.sleep(10)
            candle_writer.commit()
    except KeyboardInterrupt as e:
        logging.debug(str(e) + '\n' + 'Exiting...')
        client.stop()
