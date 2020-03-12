import numpy as np


class RecursiveLS():
    def __init__(self, steps=4, lamb=0.1):
        self.w = np.array([0]*steps)
        #self.w = np.random.randn(steps)
        self.P = np.eye(steps) * 100
        self.lamb = lamb
        self.step = steps


    def recusive_update(self, X, Z):
        '''

        :param X: 观察值 数组 n×m
        :param Z: 预测值 数组 n
        '''
        X = np.array(X)
        Z = np.array(Z)

        for x, z in zip(X, Z):
            k = np.dot(self.P, x.T) / self.lamb
            tmp = 1 + np.dot(np.dot(x, self.P), x.T)/self.lamb
            k /= tmp

            a = z - np.dot(x, self.w.T)

            self.w = self.w + k.T*a

            self.P = self.P/self.lamb - np.dot(np.dot(k, x), self.P)/self.lamb


    def predict(self, X):
        '''
        :param X: 观察值 数组 n×m
        :return:  预测值  数组 n
        '''
        y_pre = []
        for i, x in enumerate(X):
            y_pre.append(np.dot(x, self.w.T))
            if i < len(X)-1:
                z = X[i+1][-1]
                k = np.dot(self.P, x.T) / self.lamb
                tmp = 1 + np.dot(np.dot(x, self.P), x.T) / self.lamb
                k /= tmp

                a = z - np.dot(x, self.w.T)

                self.w = self.w + k.T * a

                self.P = self.P / self.lamb - np.dot(np.dot(k, x), self.P) / self.lamb

        return np.array(y_pre)

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





