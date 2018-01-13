import datetime
from btfxwss.data.model import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo=True)
engine.connect()
Session = sessionmaker(bind=engine)
session = Session()

new_asset = Trade(id=1, symbol='BTCUSD', timestamp=datetime.date(2017, 12, 15), amount=100, price=14000)
session.add(new_asset)
session.commit()
