from zipline.data.bundles import load, register
from zipline.data.data_portal import DataPortal
from zipline.utils.calendars import get_calendar
from zipline.data.bundles import quandl

import pandas as pd

quandl_bundle = load('quantopian-quandl')
start_dt = pd.Timestamp('2016-12-1', tz = 'utc')
end_dt = pd.Timestamp('2017-12-10', tz = 'utc')
quandl_data = DataPortal(quandl_bundle.asset_finder, get_calendar('NYSE'),
                       quandl_bundle.equity_daily_bar_reader.first_trading_day,
                       equity_minute_reader=quandl_bundle.equity_minute_bar_reader,
                       equity_daily_reader=quandl_bundle.equity_daily_bar_reader,
                       adjustment_reader=quandl_bundle.adjustment_reader)
asset = quandl_bundle.asset_finder.retrieve_asset(1)
df = quandl_data.get_history_window([asset],end_dt,10,'1d','close', 'daily')

print (df)