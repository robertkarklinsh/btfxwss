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

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    symbol = Column(String, ForeignKey('pairs.symbol'))
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    def __repr__(self):
        return "<OHLCV(timestamp='%s', symbol='%s', open='%s', high='%s', low='%s', close='%s', volume='%s')>" % (
            self.timestamp, self.symbol, self.open, self.high, self.low, self.close, self.volume)
