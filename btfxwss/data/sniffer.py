import logging
import time
import numpy as np
from btfxwss.queue_processor import QueueProcessor
from btfxwss.client import BtfxWss
from sqlalchemy import create_engine
from btfxwss.data.candle_writer import CandleWriter

logging.basicConfig(filename='test.log', level=logging.DEBUG)
log = logging.getLogger(__name__)


class Sniffer(QueueProcessor):
    def __init__(self, candle_writer, data_q):
        super().__init__(data_q)
        self.candle_writer = candle_writer

    def _handle_candles(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        payload = np.array(payload)
        # check for data containing single entity:
        if len(payload.shape) != 2:
            payload = payload.reshape((1,-1))
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        self.candle_writer.write(symbol, payload, local=True)


if __name__ == "__main__":
    # engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
    engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo=True)

    candle_writer = CandleWriter(engine)
    time.sleep(1)
    sniffer = Sniffer(candle_writer, None)
    client = BtfxWss(queue_processor=sniffer, log_level=logging.DEBUG)
    client.start()
    time.sleep(1)
    client.subscribe_to_candles('BTCUSD')
    client.subscribe_to_candles('ETHUSD')
    client.subscribe_to_candles('OMGUSD')
    client.subscribe_to_candles('IOTAUSD')
    try:
        while True:
            time.sleep(10)
            client.queue_processor.candle_writer.commit()
    except KeyboardInterrupt as e:
        logging.debug(str(e) + '\n' + 'Exiting...')
        client.stop()
