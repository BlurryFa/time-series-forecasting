from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from pandas import DataFrame
from data_reading import data_reading
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import FastICA


import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

def de_seasonality(df):
    after_deseasonality = {}
    seasonality_df = {}
    for name in df.columns:
        ts = df[name]
        try:
            decomposition = seasonal_decompose(ts)
        except Exception as e:
            continue
        seasonality = np.array(decomposition.seasonal.values)
        values = np.array(ts.values)
        values = values - seasonality
        after_deseasonality[name] = values
        seasonality_df[name] = seasonality

    return  DataFrame(after_deseasonality, index=df.index), DataFrame(seasonality_df, index=df.index)

if __name__ == '__main__':
    df = data_reading()
    fonsize = {'fontsize': 'xx-large',
 'fontweight': 'black',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}

    #print(rcParams['axes.titlesize'])
    df, se = de_seasonality(df)
    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.plot(df['扬州市秦邮特种金属材料有限公司'], label='De-seasonality Data')
    plt.grid()
    plt.title('De-seasonality Data',fontdict=fonsize)
    plt.subplot(212)
    plt.plot(se['扬州市秦邮特种金属材料有限公司'], label='Seasonality')
    plt.grid()
    plt.title('Seasonality', fontdict=fonsize)
    plt.savefig('./seasonal_adjustment/扬州市秦邮特种金属材料有限公司.jpg')
    plt.show()
