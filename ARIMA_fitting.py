from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels
from pandas import Series
from industrial_company import *
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
import math
import matplotlib.pyplot as plt
from season_decomposition import de_seasonality
import numpy as np
import pickle as pk
import matplotlib         ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']


class ARIMATS():

    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize_and_fit(self, time_series, order=(4,1,1)):
        if self.regressor is None:
            self.regressor = ARIMA(time_series, order).fit()
        return

    def train_test_split(self, time_series):
        train = time_series[:-2]
        test = time_series[-2:]
        return train, test

    def predict(self):
        pre_ts = self.regressor.predict(start='2016-05-01', end='2018-10-01')
        return pre_ts

    def aic_order_determine(self, time_series):
        orders = statsmodels.tsa.stattools.arma_order_select_ic(time_series, max_ar=4, max_ma=5, ic='aic')[
            'aic_min_order']
        return orders

    def plot_acf_pacf(self, time_series):
        #plt.figure(figsize=(12, 12))
        plot_acf(time_series)
        plt.savefig('C:\\Users\\Administrator\\Desktop\\seuthesis-master\\acfpacf\\acf.jpg')
        plot_pacf(time_series)
        plt.savefig('C:\\Users\\Administrator\\Desktop\\seuthesis-master\\acfpacf\\pacf.jpg')
        return

    def get_residual(self):
        residual = Series(self.regressor.resid)
        return residual

if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0] * 34, index=df.index)
    for enterprise in electronic:
        time_series = time_series + df[enterprise]
    # time_series = time_series.diff(1)
    # time_series = time_series.dropna()
    #print(time_series)
    time_series = df['扬州市秦邮特种金属材料有限公司']
    regressor = ARIMATS()
    #time_series_d1 = time_series.diff(1)
    #time_series_d1 = time_series_d1.dropna()
    #time_series_log = log_transform(time_series)
    #regressor.plot_acf_pacf(time_series_log)
    #plt.show()
    #time_series, max_value, min_value = standard_scale(time_series)
    # if is_stable(time_series):
    #     train, test = regressor.train_test_split(time_series)
    #     regressor.initialize_and_fit(train, order=(4, 0, 1))
    #     pre_ts= regressor.predict()
    #     print(sMAPE(pre_ts, test))
    # else:
    time_series_log = log_transform(time_series)
    train, test = regressor.train_test_split(time_series)
    train = log_transform(train)
    regressor.initialize_and_fit(train, order=(4, 0, 1))
    pre_ts = regressor.predict()
    pre_values = pre_ts.values
    #test_values = test.values

    for i in range(len(pre_values)):
        pre_values[i] = math.exp(pre_values[i])

    index = pre_ts.index
    pre_ts = Series(pre_values, index=index)
    #test = Series(test_values, index=index)
    plt.figure(figsize=(6, 6))
    pre_ts.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual = time_series - pre_ts
    # with open('./residual_pickle/arima_oil_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)
    plt.title('ARIMA')
    plt.grid(which='both')
    plt.savefig('./pre_plot/ARIMA.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual.plot(label='residual for ARIMA', legend=True)
    plt.grid(which='both')
    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for ARIMA', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/ARIMAr.jpg')
    plt.show()
    # print(sMAPE(pre_ts, test))
    #
    # print(pre_values)
    # print(test.values)
    # print(pre_ts)
    # print(test)
    # print(sMAPE(pre_ts, test))
    # print(MSE(pre_ts, test))










