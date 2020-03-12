from recursive_ls import RecursiveLS
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from industrial_company import *
from data_evaluating import sMAPE, MSE
import seaborn
from season_decomposition import de_seasonality
import pickle as pk

import matplotlib.pyplot as plt
import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']



if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    #print(df.cov())
    #seaborn.heatmap(df.cov())
    #plt.show()
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0]*34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]

    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()
    time_series = df['扬州市秦邮特种金属材料有限公司']
    regressor = RecursiveLS(4, 0.1)
    train, test = regressor.train_test_split(time_series_df)
    train_X, train_y = regressor.split_sequence(train)
    test_X, test_y = regressor.split_sequence(test)

    time_series_X, _ = regressor.split_sequence(time_series)
    regressor.recusive_update(train_X, train_y)

    #y_pre = regressor.predict(test_X)
    time_series_pre = regressor.predict(time_series_X)

    index = test[-2:].index
    index_plot = time_series.index[4:]
    pre_plot = Series(time_series_pre, index=index_plot)
    # values = time_series.values
    # tmp_1 = values[-3] + y_pre[0]
    # tmp_2 = values[-2] + y_pre[1]
    # y_pre = Series([tmp_1, tmp_2], index=index)
    #
    # tmp_1 = values[-3] + test_y[0]
    # tmp_2 = values[-2] + test_y[1]
    # test_y = Series([tmp_1, tmp_2], index=index)
    plt.figure(figsize=(6, 6))

    pre_plot.plot(color='green', label='Predicts', legend=True)#, label='Predicts'
    time_series.plot(color='blue', label='Original', legend=True)#, label='Original'
    residual = time_series - pre_plot
    # with open('./residual_pickle/rls_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)

    plt.title('recursive least square')
    plt.grid(which='both')
    plt.savefig('./pre_plot/rls.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual.plot(label='residual for recursive least square', legend=True)
    plt.grid(which='both')
    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for recursive least square', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/rlsr.jpg')
    plt.show()

    # y_pre = Series(y_pre, index=index)
    # test_y = Series(test_y, index=index)
    # print(sMAPE(y_pre, test_y))
    # print(MSE(y_pre, test_y))


