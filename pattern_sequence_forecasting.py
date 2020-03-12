from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas import DataFrame, Series
import pandas as pd
import numpy as np


class PSF():
    def __init__(self, time_series, k=None, w=None):
        '''
        :param time_series: 时间序列 pandas.series()
        :param k: 聚类个数
        :param w: 窗口长度
        '''
        self.w = w
        self.ts = time_series
        self.k = k
        self.data = list(time_series.values)

    def choice_best_w_k(self):
        best_w = 14
        best_w_error = 10000
        for w in reversed(range(2, 15)):
            X = []
            y = []
            y_test = self.data[-1]
            X_test = self.data[-w - 1:-1]
            for i in range(len(self.data)):

                end_ix = i + w
                if end_ix > len(self.data) - 3:
                    break
                X.append(self.data[i:end_ix])
                y.append(self.data[end_ix])
                X = np.array(X)
                y = np.array(y)

            best_k = 5
            best_k_ss = -1
            for k in range(5, 18):
                kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
                labels = kmeans.labels_
                ss = silhouette_score(X, labels, metric='euclidean')
                if ss > best_k_ss:
                    best_k = k
                    best_k_ss = ss

            kmeans = KMeans(n_clusters=best_k, random_state=1).fit(X)
            labels = kmeans.labels_
            y_pre_label = np.squeeze(kmeans.predict([X_test]))
            y_pre = np.mean(y[labels == y_pre_label])
            errors = abs(y_pre-y_test)
            if errors < best_w_error:
                best_w = w
                best_w_error = errors
                final_k = best_k

        self.w = best_w
        self.k = final_k
        return 0


    def predict(self):
        X = []
        y = []
        X_test = self.data[-self.w:]

        for i in range(len(self.data)):
            end_ix = i + self.w
            if end_ix > len(self.data) - 2:
                break
            X.append(self.data[i:end_ix])
            y.append(self.data[end_ix])
            X = np.array(X)
            y = np.array(y)
        kmeans = KMeans(n_clusters=self.k, random_state=1).fit(X)
        labels = kmeans.labels_
        label = np.squeeze(kmeans.predict([X_test]))
        y_pre = np.mean(y[labels == label])

        return y_pre


