import numpy as np
from scipy import log


def logSum(loga, logb):
    if loga < logb:
        return logb + log(1 + np.exp(loga - logb))
    else:
        return loga + log(1 + np.exp(logb - loga))


