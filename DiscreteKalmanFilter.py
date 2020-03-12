import numpy as np
from utils import OnlineLinearRegression
import random


class DiscreteKalmanFilter:


    def __init__( self ):

        self.A = None
        self.P = None
        self.Q = None

        self.H = None
        self.K = None
        self.R = None

        self.olrState = OnlineLinearRegression() #状态估计
        #self.olrMeasurement = OnlineLinearRegression() #测量估计

        self.x = None
        self.z = None



    def init(self, X, Z):
        '''

        :param X: 状态矩阵 n×m
        :param Z: 观测矩阵 n×m
        '''

        for xk_1, xk in zip(X, X[1:]):
            x, y = xk_1.reshape((1, len(xk_1))), xk.reshape((1, len(xk_1)))       
            self.olrState.update(x, y)


        self.A = self.olrState.getA()#状态传输矩阵
        self.P = self.olrState.getCovarianceMatrix()

        #for x, z in zip(X, Z):
        #    a, b = x.reshape((1, len(x))), z.reshape((1, len(z)))
        #   self.olrMeasurement.update(a, b)

        self.H = np.mat(np.eye(X.shape[1]))
        self.R = np.mat(np.zeros(shape=(X.shape[1], X.shape[1])))

        n,m = self.A.shape
        self.Q = np.random.randn(n, m)*100##这里Q是正太分布的

        self.x = np.array(X[-1]).reshape((1, X.shape[1]))
        self.z = np.mean(Z, axis=0).reshape((1, Z.shape[1]))


    def update(self, z_in):
        '''
        :param z_in:  输入的最新观测 1×m
        '''

        P = np.mat(self.P)    #噪声方差
        R = np.mat(self.R)    #噪声方差
        H = np.mat(self.H)

        z = np.mat(z_in)

        x = np.mat(self.x)

        x = np.dot(x, self.A)

        tempM = (H.T * P * H) + R

        tmat = np.linalg.pinv(np.nan_to_num(tempM))   #算出广义逆矩阵


        K = P * H * tmat    #kalman的kalman增益
        print(K)

        self.x = x + ((z - (x * H)) * K.T)
        
        I = np.mat(np.eye(len(P)))

        self.P = (I - (K * H.T)) * P    #更新P值
        #print(self.x)
        return np.squeeze(np.array(self.x))[-1] + np.random.randn()*100

    def predict(self, x, z):

        self.z = z
        nextstate = np.dot(x, self.A)
        nextmeasurement = np.dot(nextstate, self.H)


        self.x = nextstate

        P = np.mat(self.P)
        Q = np.mat(self.Q)
        A = np.mat(self.A)

        self.P = (A * P * A.T) + Q

        self.olrState.update(x, nextstate)
        self.A = self.olrState.getA()

        #self.olrMeasurement.update(x, z)
        n,m = self.A.shape
        self.Q = np.random.rand(n, m)
        #self.H = self.olrMeasurement.getA( )

        return nextstate


