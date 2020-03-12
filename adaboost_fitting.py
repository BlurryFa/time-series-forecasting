import numpy as np
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pydotplus
from sklearn import tree
from IPython.display import Image
from season_decomposition import de_seasonality
from industrial_company import *

class AdaBoostRegressorTS():

    def __init__(self, step=4):
        self.regressor = None
        self.step = step

    def initialize(self, max_depth=10, n_estimators=150):
        if self.regressor is None:
            base_estimator = DecisionTreeRegressor(max_depth=max_depth)
            self.regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators)
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

    def plot_model(self):
        for index, model in enumerate(self.regressor.estimators_):
            filename = 'decision_tree/adaboost' + str(index) + '.jpg'
            dot_data = tree.export_graphviz(model, out_file=None,
                                            feature_names=['4 days ago', '3 days ago', '2 days ago', '1 days ago'],
                                            class_names='预测值',
                                            filled=True, rounded=True,
                                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            # 使用ipython的终端jupyter notebook显示。
            Image(graph.create_png())
            graph.write_jpg(filename)




if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0] * 34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]


    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()
    regressor = AdaBoostRegressorTS()
    regressor.initialize()
    train, test = regressor.train_test_split(time_series_df)
    train_X, train_y = regressor.split_sequence(train)
    test_X, test_y = regressor.split_sequence(test)
    regressor.train(train_X, train_y)
    #regressor.plot_model()
    y_pre = regressor.predict(test_X)
    index = test[-2:].index
    #y_pre = Series(y_pre, index=index)
    #test_y = Series(test_y, index=index)
    values = time_series.values
    tmp_1 = values[-3] + y_pre[0]
    tmp_2 = values[-2] + y_pre[1]
    y_pre = Series([tmp_1, tmp_2], index=index)

    tmp_1 = values[-3] + test_y[0]
    tmp_2 = values[-2] + test_y[1]
    test_y = Series([tmp_1, tmp_2], index=index)



    print(sMAPE(y_pre, test_y))
    print(MSE(y_pre, test_y))