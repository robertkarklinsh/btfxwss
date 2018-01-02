import numpy as np
from btfxwss.algo.rapid_move import RapidMove

alg = RapidMove(1.2, 5)

print(alg.trigger_condition('btcusd', np.array([1, 1, 1, 1, 1])))
print(alg.trigger_condition('iotusd', np.array([1, 0.8, 1, 1, 1])))
print(alg.trigger_condition('zecusd', np.array([1, 1.2, 1, 1, 1])))
