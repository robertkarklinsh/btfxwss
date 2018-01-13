import numpy as np
from collections import defaultdict


class RapidMove:
    def __init__(self, change_threshold, ohlcv_array_length):
        self.change_threshold = change_threshold
        self.ohlcv_array_length = ohlcv_array_length
        self.ohlcv_array = defaultdict(self.array_factory)

    def array_factory(self):
        return np.zeros((self.ohlcv_array_length, 6))

    def trigger_condition(self, symbol, ohlcv):
        self.ohlcv_array[symbol] = np.roll(self.ohlcv_array[symbol], -1, axis=0)
        self.ohlcv_array[symbol][-1] = ohlcv
        non_zero_array = self.ohlcv_array[symbol][np.nonzero(self.ohlcv_array[symbol][:, 2])]
        res = [False, False]
        if non_zero_array[-1, 2] * self.change_threshold < np.amax(non_zero_array[:, 2]):
            res[0] = True
        if non_zero_array[-1, 2] / self.change_threshold > np.amin(non_zero_array[:, 2]):
            res[1] = True
        return res

