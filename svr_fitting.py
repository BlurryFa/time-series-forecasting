import numpy as np
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
from industrial_company import *
import pickle as pk
from season_decomposition import de_seasonality
from sklearn.svm import SVR
import matplotlib.pyplot as plt

class SVRTS():

    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self):
        if self.regressor is None:
            self.regressor = SVR(kernel='linear')
        return

    def train(self, X, y):
        self.regressor.fit(X, y)
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
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre


if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0] * 34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]

    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()
    time_series = df['扬州市秦邮特种金属材料有限公司']
    # time_series = time_series.diff(1)
    # time_series = time_series.dropna()
    regressor = SVRTS()
    regressor.initialize()
    train, test = regressor.train_test_split(time_series)
    train_X, train_y = regressor.split_sequence(train)
    test_X, test_y = regressor.split_sequence(test)
    time_series_plot, _ = regressor.split_sequence(time_series)

    regressor.train(train_X, train_y)
    y_pre = regressor.predict(time_series_plot)
    index = time_series.index[4:]
    y_pre = Series(y_pre, index=index)
    # values = time_series.values
    # tmp_1 = values[-3] + y_pre[0]
    # tmp_2 = values[-2] + y_pre[1]
    # y_pre = Series([tmp_1, tmp_2], index=index)
    #
    # tmp_1 = values[-3] + test_y[0]
    # tmp_2 = values[-2] + test_y[1]
    # test_y = Series([tmp_1, tmp_2], index=index)

    plt.figure(figsize=(6, 6))
    y_pre.plot(color='green', label='Predicts', legend=True)  # , label='Predicts'
    time_series.plot(color='blue', label='Original', legend=True)  # , label='Original'
    residual = time_series - y_pre
    # with open('./residual_pickle/svr_manufacture_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)
    plt.title('support vector regression')
    plt.grid(which='both')
    plt.savefig('./pre_plot/svr.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual.plot(label='residual for support vector regression', legend=True)
    plt.grid(which='both')
    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for support vector regression', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/svrr.jpg')
    plt.show()

    #y_pre = Series(y_pre, index=index)
    # print(sMAPE(y_pre, test_y))
    # print(MSE(y_pre, test_y))





