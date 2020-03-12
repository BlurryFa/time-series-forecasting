from keras.layers import Dense
from keras.models import Sequential
import numpy as np


class AnnTS(object):
    def __init__(self, predictors):
        self.regressor = None
        self.predicors = predictors

    def initialize(self):
        if self.regressor is None:
            self.regressor = Sequential()
            self.regressor.add(Dense(50, activation='relu', input_shape=(self.predict,), use_bias=True))
            self.regressor.add(Dense(30, activation='relu', use_bias=True))
            self.regressor.add(Dense(1))
            self.regressor.compile(optimizer='adam', loss='mse')
        return

    def train(self, X, y):
        self.regressor.fit(X, y, epochs=1500, verbose=0)
        return

    def train_test_split(self, X, y):
        train_X = X[:-1]
        test_X = X[-1]
        train_y = y[:-1]
        test_y = y[-1]

        return train_X, test_X, train_y, test_y

    def predict(self, X):
        y_pre = self.regressor.predict(X)
        y_pre = np.squeeze(y_pre)
        return y_pre