import datetime
from btfxwss.data.model import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine


engine = create_engine('sqlite:////home/robert/PyProjects/btfxwss/tests/foo.bd', echo = True)
engine.connect()
Base.metadata.create_all(engine)




