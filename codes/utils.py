import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    smap = smape_arr(y_true, y_pred)
    print(smap)
    
    return np.mean(smap)

def smape_arr(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = (num[pos_ind] / dem[pos_ind]) * 100

    return smap