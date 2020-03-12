from statsmodels.tsa.holtwinters import Holt
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from industrial_company import *
import pickle as pk
import matplotlib.pyplot as plt
from season_decomposition import de_seasonality
import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']


def ses_fitting(time_series):
    train = time_series[:-2]
    test = time_series[-2:]
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()
    residual = Series(model.resid)
    return model, test, residual

def holt_fitting(time_series):
    train = time_series[:-2]
    test = time_series[-2:]
    model = Holt(train, damped=True).fit()
    residual = Series(model.resid)
    return model, test, residual

if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0] * 34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]

    # time_series = time_series.diff(1)
    # time_series = time_series.dropna()


    # bias = Series([2000] * 33, index=time_series.index)
    # time_series = time_series + bias
    time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series_log = log_transform(time_series)

    model_h, test_h, residual = holt_fitting(time_series_log)
    model, test, residual = ses_fitting(time_series_log)
    pre_ts_h = model_h.predict(start='2016-05-01', end='2018-10-01')
    pre_ts = model.predict(start='2016-05-01', end='2018-10-01')
    pre_values = pre_ts.values
    pre_values_h = pre_ts_h.values
    #test_values_h = test_h.values

    for i in range(len(pre_values)):
        pre_values[i] = math.exp(pre_values[i])
        pre_values_h[i] = math.exp(pre_values_h[i])
        #test_values_h[i] = math.exp(test_values_h[i])
        #test_values[i] = math.exp(test_values[i])

    index = pre_ts.index
    pre_ts_h = Series(pre_values_h, index=index)
    #test_h = Series(test_values_h, index=index)
    #test_values = Series(test_values, index=index)
    pre_ts = Series(pre_values, index=index)
    # print(sMAPE(pre_ts, test_h))
    # print(MSE(pre_ts, test_h))
    # print(sMAPE(pre_ts_h, test_h))
    # print(MSE(pre_ts_h, test_h))


    #test = Series(test_values, index=index)
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    pre_ts_h.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_h = time_series - pre_ts_h
    # with open('./residual_pickle/holt_oil_df.pkl', 'wb') as f:
    #     pk.dump(residual_h, f)
    plt.title('Holt-Winters')
    plt.grid(which='both')
    plt.subplot(212)
    pre_ts.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual = time_series - pre_ts
    # with open('./residual_pickle/es_oil_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)
    plt.title('Simple Exponetial Smooth')
    plt.grid(which='both')
    plt.savefig('./pre_plot/es.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual_h.plot(label='residual for Holt-Winters', legend=True)
    residual.plot(label='residual for SES', legend=True)
    plt.grid(which='both')

    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for SES', legend=True)
    residual_h.plot(kind='kde', label='residual kernel density for Holt-Winters', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/esr.jpg')
    plt.show()



