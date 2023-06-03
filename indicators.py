import pandas as pd
import numpy as np
import scipy
import math
from reversibility import perm_ts_reversibility,relative_async_index
import matplotlib.pyplot as plt

def rw_ptsr(arr: np.array, lookback: int):
    # Rolling window permutation time series reversibility
    rev = np.zeros(len(arr))
    rev[:] = np.nan
    
    lookback_ = lookback + 2
    for i in range(lookback_, len(arr)):
        dat = arr[i - lookback_ + 1: i+1]
        rev_w = perm_ts_reversibility(dat) 

        if np.isnan(rev_w):
            rev[i] = rev[i - 1]
        else:
            rev[i] = rev_w

    return rev





def rw_rai(arr: np.array, lookback: int):
    # Rolling window relative asynchronous index
    rev = np.zeros(len(arr))
    rev[:] = np.nan
    
    for i in range(lookback, len(arr)):
        dat = arr[i - lookback + 1: i+1]
        rev_w = relative_async_index(dat) 
        rev[i] = rev_w

    return rev


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    
    # You could use changes in price instead.  
    data['d'] = np.log(data['close']).diff()

    lb = 365
    data['rai'] =  rw_rai(data['close'].to_numpy(), lb)
    data['ptsr'] =  rw_ptsr(data['close'].to_numpy(), lb)
    data['rai_s'] = data['rai'].ewm(7).mean()
    

    plt.style.use('dark_background')
    np.log(data['close']).plot()
    plt.twinx()
    data['ptsr'].plot(color='orange', label='PTSR365')
    #data['rai'].plot(color='orange', label='RAI365')
    data['rai'].ewm(7).mean().plot(color='red', label='RAI365 - EMA7')
    plt.legend()


