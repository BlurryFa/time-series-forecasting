import numpy as np
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
from xgboost_fitting import XGboostTS
import pickle as pk
from lstm_fitting import Stack_Lstm
from sklearn.linear_model import LinearRegression
from season_decomposition import de_seasonality
from industrial_company import *

import matplotlib.pyplot as plt
import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']


def split_sequence(time_series, steps=4):
    X, y = list(), list()
    values = time_series.values
    for i in range(len(values)):
        end_ix = i + steps
        if end_ix > len(values) - 1:
            break
        seq_x, seq_y = values[i:end_ix], values[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def train_test_split(time_series):
    train = time_series[:-2]
    test = time_series[-6:]
    return train, test

###   theta分解

def theta_transforming(time_series, theta):
    '''

    :param time_series:  输入时间序列 pandas.Series()
    :param theta: Theta值 浮点数
    :return: Theta线 pandas.Series()
    '''
    values = time_series.values
    x_1 = values[0]
    x_2 = values[1]
    n = len(values)
    n_double = np.double(n)

    time_series_1 = time_series.diff(1)
    time_series_1 = time_series_1.dropna()
    time_series_2 = time_series_1.diff(1)
    time_series_2 = time_series_2.dropna()

    curvatures = list(time_series_2.values)

    constant_1 = 0.0

    for i in range(3, n+1):
        for t in range(2, i):
            constant_1 += np.double((i-t)) * curvatures[t-2]

    constant_2 = 0.0

    for i in range(3, n+1):
        temp = 0.0
        for t in range(2, i):
            temp += np.double((i-t)) * curvatures[t-2]
        constant_2 += (i-1) * temp

    y_1 = x_1 + 6*(theta-1)*constant_2/(n*(n+1)) + 2*(2*n-1)*(1-theta)*constant_1/(n*(n+1))

    y_2 = (n-3)*y_1/(n-1) + x_2 + (3-n)*x_1/(n-1) + 2*(1-theta)*constant_1/((n-1)*n)

    y_list = [0]*n

    y_list[0] = y_1
    y_list[1] = y_2

    for i in range(3, n+1):
        temp = 0.0
        for t in range(2, i):
            temp += (i-t)*curvatures[t-2]
        y_list[i-1] = y_1 + (i-1)*(y_2-y_1) + theta*temp
    result = Series(y_list, index=time_series.index)
    return result

if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0] * 34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]
    time_series = df['扬州市秦邮特种金属材料有限公司']
    #print(time_series)
    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()

    plt.figure(figsize=(6, 6))
    plt.plot(time_series.values, color='b', marker='o', label='Original', )
    theta_01 = theta_transforming(time_series, 0.1)
    plt.plot(theta_01.values, color='g', marker='x', label='Theta=0.1')
    theta_05 = theta_transforming(time_series, 0.5)
    plt.plot(theta_05.values, color='r', marker='.', label='Theta=0.5')
    theta_15 = theta_transforming(time_series, 1.5)
    theta_19 = theta_transforming(time_series, 1.9)
    plt.plot(theta_15.values, color='c', marker='>', label='Theta=1.5')
    plt.plot(theta_19.values, color='m', marker='<', label='Theta=1.9')
    plt.legend(loc='upper right')
    plt.xticks([0, 6, 12, 18, 24, 30], [df.index.to_period()[i] for i in [0, 6, 12, 18, 24, 30]])
    plt.title('Theta decomposition')
    plt.grid(which='both')
    plt.savefig('./pre_plot/thetadecomposition1.jpg')
    plt.show()

    train, test = train_test_split(time_series)
    train_X, train_y = split_sequence(time_series)
    test_X, test_y = split_sequence(test)

    train_01, test_01 = train_test_split(theta_01)
    train_X_01, train_y_01 = split_sequence(theta_01)
    test_X_01, test_y_01 = split_sequence(test_01)

    train_19, test_19 = train_test_split(theta_19)
    train_X_19, train_y_19 = split_sequence(theta_19)
    test_X_19, test_y_19 = split_sequence(test_19)

    stack_lstm = Stack_Lstm()
    stack_lstm.initialize()
    stack_lstm.train(train_X_19, train_y_19)

    linearregressor = LinearRegression()
    linearregressor.fit(train_X_01, train_y_01)


    theta_19_pre = np.squeeze(stack_lstm.predict(train_X_19))
    theta_01_pre = np.squeeze(linearregressor.predict(train_X_01))
    y_pre = (theta_19_pre + theta_01_pre)/2

    index = time_series.index[4:]
    y_pre = Series(y_pre, index)

    #test_y = time_series[-2:]

    # print(sMAPE(y_pre, test_y))
    # print(MSE(y_pre, test_y))

    theta_19_pre = Series(theta_19_pre, index=index)
    theta_01_pre = Series(theta_01_pre, index=index)
    y_pre = Series(y_pre, index=index)

    plt.figure(figsize=(6, 6))
    theta_19_pre.plot(color='g', label='Theta=1.9 predicts', legend=True)
    theta_19.plot(color='r', label='Theta=1.9', legend=True)
    theta_01_pre.plot(color='c', label='Theta=0.1 predicts', legend=True)
    theta_01.plot(color='m', label='Theta=0.1', legend=True)
    y_pre.plot(color='black', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual = time_series - y_pre
    # with open('./residual_pickle/theta_vehicle_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)
    plt.title('Theta')
    plt.grid(which='both')
    plt.savefig('./pre_plot/Theta.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual.plot(label='residual for Theta', legend=True)
    plt.grid(which='both')
    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for Theta', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/Thetar.jpg')
    plt.show()





