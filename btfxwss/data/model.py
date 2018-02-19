import numpy as np
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Date, DateTime, Integer, String

Base = declarative_base()


class Pair(Base):
    __tablename__ = 'pairs'

    sid = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True)
    start_date = Column(Date)

    def __repr__(self):
        return "<Pair(symbol='%s', start_date='%s')>" % (
            self.symbol, self.start_date)


class OHLCV(Base):
    __tablename__ = 'ohlcv'

    timestamp = Column(DateTime, primary_key=True)
    symbol = Column(String, ForeignKey('pairs.symbol'))
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    def __repr__(self):
        return "<OHLCV(id = '%s', timestamp='%s', symbol='%s', open='%s', high='%s', low='%s', close='%s', volume='%s')>" % (
            self.id, self.timestamp, self.symbol, self.open, self.high, self.low, self.close, self.volume)


class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    symbol = Column(String, ForeignKey('pairs.symbol'), primary_key=True)
    amount = Column(Float)
    price = Column(Float)

    def __repr__(self):
        return "<Trades(id='%s', timestamp='%s', symbol='%s', amount='%s', price='%s')>" % (
            self.id, self.timestamp, self.symbol, self.amount, self.price)
