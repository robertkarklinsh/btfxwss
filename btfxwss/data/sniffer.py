import logging
import time
import numpy as np
from collections import defaultdict
from btfxwss.queue_processor import QueueProcessor
from btfxwss.client import BtfxWss
from sqlalchemy import create_engine
from btfxwss.data.candle_writer import CandleWriter

logging.basicConfig(filename='test.log', level=logging.DEBUG)
log = logging.getLogger(__name__)


class Sniffer(QueueProcessor):
    def __init__(self, data_q=None):
        super().__init__(data_q)
        self.candle_handlers = []
        self.timestamp_index = defaultdict(int)

    def _handle_candles(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        if payload.size == 0:
            return
        # check for data containing single entity:
        if len(payload.shape) != 2:
            if self.timestamp_index[symbol] < payload[0]:
                self.timestamp_index[symbol] = payload[0]
                payload = payload.reshape((1, -1))
            else:
                return
        # formatting payload to [ts, o, h, l, c, v]
        payload = payload[:, [0, 1, 3, 4, 2, 5]]

        for handler in self.candle_handlers:
            handler(symbol, payload)


if __name__ == "__main__":
    # engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
    engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo=True)

    candle_writer = CandleWriter(engine)
    time.sleep(1)
    sniffer = Sniffer()
    sniffer.candle_handlers.append(candle_writer.write)


    client = BtfxWss(queue_processor=sniffer, log_level=logging.DEBUG)
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
    #        candle_writer.commit()
    except KeyboardInterrupt as e:
        logging.debug(str(e) + '\n' + 'Exiting...')
        client.stop()
