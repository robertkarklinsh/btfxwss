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


    def write(self, symbol, data, local=False):
        '''
        :param symbol: string
        :param data: [milliseconds since Epoch (integer), open (float), close (float), high (float), low (float), volume (float)]
        :param local:
        :return:
        '''

        # Check for sqllite
        for chunk in data:
            if local is True:
                timestamp = datetime.datetime.fromtimestamp(chunk[0] / 1000)

            new_ohlcv = OHLCV(timestamp=timestamp, symbol=symbol, open=chunk[1], high=chunk[3],
                          low=chunk[4], close=chunk[2], volume=chunk[5])

            self.session.add(new_ohlcv)

    def commit(self):
        try:
            self.session.commit()
        except Exception as e:
            logger.error("Couldn't commit to db" + "\n" + str(e))
