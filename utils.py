import numpy as np

def moving_average(data, period):
    mavg = []
    for i in range(0, len(data)):
        if i < period:
            mavg.append(np.nan)
        else:
            mavg.append(np.average(data[i-period:i]))
    return np.array(mavg)