import pandas as pd
import numpy as np


class OnlineLinearRegression:


    def __init__(self):

        self.error = None
        self.theta = None
        self.cov = None
        self.isTrained = False


    def update(self, x, y):

        self.priorY = y
        self.priorX = x

        xm = np.mat (x)
        ym = np.mat (y)

        all_zeros = not np.any(self.error)
        if all_zeros:
            self.error = np.zeros(y.shape)

        all_zeros = not np.any(self.theta)

        if all_zeros:

            tmat = np.linalg.pinv(np.nan_to_num(xm.T * xm))#广义逆矩阵
            self.theta = (tmat) * xm.T * ym
            self.isTrained = True

        all_zeros = not np.any(self.cov)

        if all_zeros:
            self.cov = np.dot(x.T, x)

        if not self.isTrained:

            cov = np.mat(self.cov)

            theta = np.mat(xm * self.theta)

            self.error = ym - theta

            Im = np.mat(np.eye(x.shape[1]))

            self.cov = cov * np.mat(Im - ((xm.T * xm * cov) / (1 + (xm * cov * xm.T))))#更新方差

            self.theta = theta + (self.cov * xm.T * self.error)

            self.isTrained = False


    def getA(self):
        # 返回估计的传输矩阵
        return self.theta



    def getCovarianceMatrix(self):
        # 返回误差的协方差矩阵

        theta = np.mat(self.getA())
        Xm = np.mat(self.priorX)

        ypost = Xm * theta
        yprior = self.priorY
        error = ypost - yprior
        return np.dot(error.T, error)







