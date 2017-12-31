import datetime
from btfxwss.data.model import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


engine = create_engine('postgresql://robertkarklinsh:onnkzuha26@35.198.127.49:5432/bitfinex', echo = True)
engine.connect()
Base.metadata.create_all(engine)




