import sched, time

from btfxwss.client import BtfxWss as Client


class Robot(Client):
    def __init__(self, **kwargs):
        self._settings = kwargs
        self._s = sched.scheduler(time.time, time.sleep)

        Client.__init__(self, self._settings['API_KEY'], self._settings['API_SECRET'])

        self._refresh_time = self._settings['REFRESH_TIME']
        self._buy_window = self._settings['BUY_WINDOW']
        self._sell_window = self._settings['BUY_WINDOW']
        self._buy_roc = self._settings['BUY_ROC']
        self._sell_position_roc = self._settings['SELL_POSITION_ROC']

        self.start()
        self.authenticate()
        self.subscribe_to_trades('tIOTBTC')

    def _roc_simple_alg(self):
        # print self.ticker('IOTBTC')
        # print time.time()
        # print self.past_trades('IOTBTC', {'timestamp': str(time.time() - self._buy_window), 'limit_trades': 5})
        #      past_price = self.past_trades(
        #     'tIOTBTC',
        #     {
        #         'limit': 1,
        #         'start': int(round((time.time() - self._buy_window) * 1000)),
        #         'end': int(round(time.time() * 1000)),
        #         'sort': 1
        #     })#, dtype='int8,int8,float32,float32')
        # print past_price
        # self._s.enter(5, 1, self._roc_simple_alg, ())

        # self.subscribe_to_trades('tIOTBTC')
        self.subscribe_to_trades('tBTCUSD')
        data = self.trades('tBTCUSD').get()
        print(data)
        self._s.enter(5, 1, self._roc_simple_alg, ())

    def run(self):
        self._s.enter(5, 1, self._roc_simple_alg, ())
        self._s.run()
