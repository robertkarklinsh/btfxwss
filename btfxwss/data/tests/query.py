import pandas as pd
import numpy as np
import time
import datetime
from btfxwss.data.model import Pair, OHLCV, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
engine.connect()
# Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
# #new_asset = Pair(symbol='BTCUSD', start_date=datetime.date(2017, 12, 15))
# new_ohlcv = OHLCV(timestamp = datetime.datetime.fromtimestamp(time.time()),
#                   symbol = 'BTCUSD',
#                   open = 16000,
#                   high = 17000,
#                   low = 15000,
#                   close = 16500,
#                   volume = 10)
# #session.add(new_asset)
# session.add(new_ohlcv)
#
# session.commit()

# query = session.query(Pair).statement
# df = pd.read_sql_query(query, engine, index_col='sid', parse_dates=['start_date'])
# df['asset_name'] = df.symbol
# df = df[['symbol','asset_name','start_date']]
# element = df['asset_name'][1]
# print(type(element))

# query = session.query(Pair).statement
# df = pd.read_sql_query(query, engine, index_col='sid', parse_dates=['start_date'])
# df['asset_name'] = df.symbol
# df = df[['symbol', 'asset_name', 'start_date']]
# print(df)



# query = session.query(OHLCV, Pair).join(Pair).statement
# df = pd.read_sql_query(query, engine, index_col='timestamp', parse_dates=['timestamp'])
# data = df.loc[:, ['sid', 'open', 'high', 'low', 'close']]
# grouped = data.groupby(['sid'])
# print (df)

#
# def _calc_minute_index(market_opens, minutes_per_day):
#     minutes = np.zeros(len(market_opens) * minutes_per_day,
#                        dtype='datetime64[ns]')
#     deltas = np.arange(0, minutes_per_day, dtype='timedelta64[m]')
#     for i, market_open in enumerate(market_opens):
#         start = market_open.asm8
#         minute_values = start + deltas
#         start_ix = minutes_per_day * i
#         end_ix = start_ix + minutes_per_day
#         minutes[start_ix:end_ix] = minute_values
#     return pd.to_datetime(minutes, utc=True, box=True)
#
# market_opens = pd.Series(np.arange(np.datetime64('2017-12-25'), np.datetime64('2017-12-29')))
# minutes_per_day = 360
# index = _calc_minute_index(market_opens=market_opens,minutes_per_day=minutes_per_day)
# query = session.query(OHLCV).statement
# df = pd.read_sql_query(query, engine, index_col='timestamp', parse_dates=['timestamp'])
# print(index[-1])
# last_minute_to_write = pd.Timestamp(df.index[-1], tz='UTC')
# # a = index.get_loc(last_minute_to_write)
# print (type(last_minute_to_write))

query = session.query(OHLCV).statement
df = pd.read_sql_query(query, engine, index_col='timestamp', parse_dates=['timestamp'])
df = df.sort_index()
print(df)



#
#
# df = pd.read_sql_query(('low','high'))
