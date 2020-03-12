from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
from industrial_company import *
import numpy as np
from season_decomposition import de_seasonality
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
from keras.utils import plot_model


class VanillaLstmTS():

    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(LSTM(50, activation='relu', input_shape=(self.step, 1)))
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.regressor.fit(X, y, epochs=1200, verbose=0)
        return

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.step
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
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


class Stack_Lstm():
    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.step, 1)))
            self.regressor.add(LSTM(50, activation='relu'))
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.regressor.fit(X, y, epochs=1200, verbose=0)
        return

    def plot_model(self):
        plot_model(self.regressor, to_file='neural_network_model/stack_lstm.pdf', show_shapes=True)

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.step
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
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre

class BidirectionalLstm():
    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=(self.step, 1)))
            self.regressor.add(Bidirectional(LSTM(50, activation='relu')))
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.regressor.fit(X, y, epochs=1200, verbose=0)
        return

    def plot_model(self):
        plot_model(self.regressor, to_file='neural_network_model/bi_lstm.pdf', show_shapes=True)

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.step
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
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


class CnnLstm():
    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                      input_shape=(2, 2, 1)))
            self.regressor.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            self.regressor.add(TimeDistributed(Flatten()))
            self.regressor.add(LSTM(50, activation='relu'))
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        X = X.reshape((X.shape[0], 2, int(self.step/2), 1))
        self.regressor.fit(X, y, epochs=1200, verbose=0)
        return

    def plot_model(self):
        plot_model(self.regressor, to_file='neural_network_model/conv1_lstm.pdf', show_shapes=True)

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.step
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
        X = X.reshape((X.shape[0], 2, int(self.step/2), 1))
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


class ConvLstm():
    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                                 input_shape=(2, 1, int(self.step/2), 1)))
            self.regressor.add(Flatten())
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        X = X.reshape((X.shape[0], 2, 1, int(self.step / 2), 1))
        self.regressor.fit(X, y, epochs=1200, verbose=0)
        return

    def plot_model(self):
        plot_model(self.regressor, to_file='neural_network_model/conv2_lstm.pdf', show_shapes=True)

    def split_sequence(self, time_series):
        X, y = list(), list()
        values = time_series.values
        for i in range(len(values)):
            end_ix = i + self.step
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
        X = X.reshape((X.shape[0], 2, 1, int(self.step/2), 1))
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0] * 34, index=df.index)
    for enterprise in construction:
        time_series = time_series + df[enterprise]

    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()
    regressor = CnnLstm()
    regressor.initialize()
    train, test = regressor.train_test_split(time_series_df)
    train_X, train_y = regressor.split_sequence(train)
    test_X, test_y = regressor.split_sequence(test)
    regressor.train(train_X, train_y)
    y_pre = regressor.predict(test_X)
    index = test[-2:].index

    values = time_series.values
    tmp_1 = values[-3] + y_pre[0]
    tmp_2 = values[-2] + y_pre[1]
    y_pre = Series([tmp_1, tmp_2], index=index)

    tmp_1 = values[-3] + test_y[0]
    tmp_2 = values[-2] + test_y[1]
    test_y = Series([tmp_1, tmp_2], index=index)
    #y_pre = Series(y_pre, index=index)
    #test_y = Series(test_y, index=index)
    print(sMAPE(y_pre, test_y))
    print(MSE(y_pre, test_y))







