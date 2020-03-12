from data_reading import data_reading
from industrial_company import *
from pandas import DataFrame, Series
import numpy as np
from sklearn.decomposition import PCA
from season_decomposition import de_seasonality
from data_preprocessing import standard_scale
import matplotlib.pyplot as plt
import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']



def pca(XMat, k):
    '''

    :param XMat: 输入矩阵， n×m
    :param k:  输出成分数
    :return:  变换后的矩阵，重构矩阵
    '''
    average = np.mean(XMat, axis=0)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        #特征向量是列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    X_vehicle = []
    X_oil = []
    X_textile = []
    X_construction = []
    X_mechanical = []
    X_electronic = []
    for index in df.index:
        tmp_1 = []
        for name in electronic:
            tmp_1.append(standard_scale(df[name])[0][index])
        X_electronic.append(tmp_1)

        tmp_2 = []
        for name in manufacture:
            tmp_2.append(standard_scale(df[name])[0][index])
        X_mechanical.append(tmp_2)

        tmp_3 = []
        for name in construction:
            tmp_3.append(standard_scale(df[name])[0][index])
        X_construction.append(tmp_3)

        tmp_4 = []
        for name in vehicle:
            tmp_4.append(standard_scale(df[name])[0][index])
        X_vehicle.append(tmp_4)

        tmp_5 = []
        for name in oil:
            tmp_5.append(standard_scale(df[name])[0][index])
        X_oil.append(tmp_5)

        tmp_6 = []
        for name in textile:
            tmp_6.append(standard_scale(df[name])[0][index])
        X_textile.append(tmp_6)

    X_mechanical = PCA(n_components=4).fit_transform(X_oil)

    plt.figure(figsize=(6, 6))

    plt.subplot(411)
    plt.plot(X_mechanical[:, 0])
    #plt.grid(which='both')
    plt.xticks([0, 6, 12, 18, 24, 30], [df.index.to_period()[i] for i in [0, 6, 12, 18, 24, 30]])
    plt.title('石化行业主成分分析')

    plt.subplot(412)
    plt.plot(X_mechanical[:, 1])
    #plt.grid(which='both')
    plt.xticks([0, 6, 12, 18, 24, 30], [df.index.to_period()[i] for i in [0, 6, 12, 18, 24, 30]])

    plt.subplot(413)
    plt.plot(X_mechanical[:, 2])
    #plt.grid(which='both')
    plt.xticks([0, 6, 12, 18, 24, 30], [df.index.to_period()[i] for i in [0, 6, 12, 18, 24, 30]])

    plt.subplot(414)
    plt.plot(X_mechanical[:, 3])
    #plt.grid(which='both')
    plt.xticks([0, 6, 12, 18, 24, 30], [df.index.to_period()[i] for i in [0, 6, 12, 18, 24, 30]])

    plt.savefig('./pca/oil.jpg')
    plt.show()

