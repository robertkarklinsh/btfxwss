import datetime

def btfx_ts_to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts/1000)




