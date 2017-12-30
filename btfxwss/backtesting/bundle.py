import zipline.data.bundles.core as bundles
import pandas as pd
import numpy as np
from btfxwss.data.model import Pair
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from btfxwss.data.model import OHLCV

engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo=True)
engine.connect()
Session = sessionmaker(bind=engine)
session = Session()


@bundles.register("bitfinex")
def bitfinex_bundle(environ,
                    asset_db_writer,
                    minute_bar_writer,
                    daily_bar_writer,
                    adjustment_writer,
                    calendar,
                    start_session,
                    end_session,
                    cache,
                    show_progress,
                    output_dir):
    '''Implementation of ingest function

    :param environ:
    :param asset_db_writer:
    :param minute_bar_writer:
    :param daily_bar_writer:
    :param adjustment_writer:
    :param calendar:
    :param start_session:
    :param end_session:
    :param cache:
    :param show_progress:
    :param output_dir:
    :return:
    '''

    query = session.query(Pair).statement
    df = pd.read_sql_query(query, engine, index_col='sid', parse_dates=['start_date'])
    df['asset_name'] = df.symbol
    df = df[['symbol', 'asset_name', 'start_date']]
    asset_db_writer.write(df)

    query = session.query(OHLCV, Pair).join(Pair).statement
    df = pd.read_sql_query(query, engine, index_col='timestamp', parse_dates=['timestamp'])
    data = df.loc[:, ['sid', 'open', 'high', 'low', 'close']]
    grouped = data.groupby(['sid'])
    data = []
    for value, group in grouped:
        data.append((value, group))

    daily_bar_writer.write(data, show_progress=show_progress)
    # adjustment_writer.write(
    #     splits=pd.concat(splits, ignore_index=True),
    #     dividends=pd.concat(dividends, ignore_index=True),
    # )
