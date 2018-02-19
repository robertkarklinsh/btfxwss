import datetime
import sys
import logging
from btfxwss.data.model import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class CandleWriter(object):
    def __init__(self, engine):
        self.engine = engine
        try:
            self.engine.connect()
        except Exception as e:
            logger.error("Can not connect to db" + "\n" + str(e))

        Session = sessionmaker(bind=engine)
        self.session = Session()

    def write(self, symbol, data):
        '''
        :param symbol: string
        :param data: [milliseconds since Epoch (integer), open (float), close (float), high (float), low (float), volume (float)]
        :return:
        '''

        # Check for sqllite
        for dt in data.index:
            # new_ohlcv = OHLCV(timestamp=timestamp, symbol=symbol, open=float(chunk[1]), high=float(chunk[3]),
            #                   low=float(chunk[4]), close=float(chunk[2]), volume=float(chunk[5]))

            new_ohlcv = OHLCV(timestamp=dt.to_pydatetime(), symbol=symbol, open=float(data.loc[dt, 'open']), high=float(data.loc[dt, 'high']),
                              low=float(data.loc[dt, 'low']), close=float(data.loc[dt, 'close']),
                              volume=float(data.loc[dt, 'volume']))
            self.session.add(new_ohlcv)

    def commit(self):
        try:
            self.session.commit()
        except Exception as e:
            logger.error("Couldn't commit to db" + "\n" + str(e))
