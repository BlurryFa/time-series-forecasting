import numpy as np
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
import math
from season_decomposition import de_seasonality
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from industrial_company import *


class AnnTS():

    def __init__(self, steps=4):
        self.regressor = None
        self.steps = steps

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(Dense(50, activation='relu', input_shape=(self.steps,), use_bias=True, name='dense_1'))
            self.regressor.add(Dense(50, activation='relu', name='dense_2'))
            self.regressor.add(Dense(30, activation='relu', use_bias=True, name='dense_3'))
            self.regressor.add(Dense(1, name='dense_4'), )
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def plot_model(self):
        plot_model(self.regressor, to_file='neural_network_model/fnn.pdf', show_shapes=True)

    def train(self, X, y):
        '''
        :param X: 二维数据 n×m
        :param y: 一维数组 n
        :return:
        '''
        self.regressor.fit(X, y, epochs=2000, verbose=0)
        return

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.steps
            if end_ix > len(values) - 1:
                break
            seq_x, seq_y = values[i:end_ix], values[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train_test_split(self, time_series):
        train = time_series[:-2]
        test = time_series[-6:]
        return train, test

    def predict(self, X):
        '''
        :param X: 二维数组 n×m
        :return: 预测值
        '''
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0] * 34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]

    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()

    regressor = AnnTS()
    regressor.initialize()

    train, test = regressor.train_test_split(time_series_df)
    train_X, train_y = regressor.split_sequence(train)
    test_X, test_y = regressor.split_sequence(test)
    regressor.train(train_X, train_y)
    regressor.plot_model()
    y_pre = regressor.predict(test_X)
    index = test[-2:].index
    values = time_series.values
    tmp_1 = values[-3] + y_pre[0]
    tmp_2 = values[-2] + y_pre[1]
    y_pre = Series([tmp_1, tmp_2], index=index)

    tmp_1 = values[-3] + test_y[0]
    tmp_2 = values[-2] + test_y[1]
    test_y = Series([tmp_1, tmp_2], index=index)
    y_pre = Series(y_pre, index=index)
    test_y = Series(test_y, index=index)
    print(sMAPE(y_pre, test_y))
    print(MSE(y_pre, test_y))