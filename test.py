from data_reading import data_reading
from industrial_company import *
from pandas import DataFrame, Series
import numpy as np
from data_preprocessing import *
from season_decomposition import de_seasonality
import matplotlib.pyplot as plt
import pickle as pk

import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == '__main__':
    with open('./residual_pickle/con1lstm_oil_df.pkl', 'rb') as f:
        conv1_oil = pk.load(f)

    plt.figure(figsize=(6, 6))
    conv1_oil.plot(kind='kde', label='Conv1 LSTM', legend=True)
    plt.grid(which='both')
    plt.title('石化产业')
    plt.savefig('./other/c1lstmoildf.jpg')
