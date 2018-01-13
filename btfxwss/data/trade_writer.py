import datetime
import sys
import logging
from btfxwss.data.model import *
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TradeWriter(object):
    def __init__(self, engine):
        self.engine = engine
        try:
            self.engine.connect()
        except Exception as e:
            logger.error("Can not connect to db" + "\n" + str(e))

        Session = sessionmaker(bind=engine)
        self.session = Session()

    def write_initial_data(self, symbol, data):
        for entity in data:
            timestamp = datetime.datetime.fromtimestamp(entity[1] / 1000)

            new_trade = Trade(id=int(entity[0]), timestamp=timestamp, symbol=symbol, amount=float(entity[2]), price=float(entity[3]))

            self.session.add(new_trade)

    def write_entity(self, symbol, entity):
        timestamp = datetime.datetime.fromtimestamp(entity[1] / 1000)

        new_trade = Trade(id=int(entity[0]), timestamp=timestamp, symbol=symbol, amount=float(entity[2]), price=float(entity[3]))

        self.session.add(new_trade)

    def commit(self):
        try:
            self.session.commit()
        except Exception as e:
            logger.error("Couldn't commit to db" + "\n" + str(e))
