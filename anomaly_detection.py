from data_reading import data_reading
from data_preprocessing import *
from sklearn.neighbors import NearestNeighbors
from industrial_company import *
import matplotlib.pyplot as plt
from season_decomposition import de_seasonality
import numpy as np
import matplotlib         ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False


def euclidean_distance(x_1, x_2):
    x_1 = np.squeeze(x_1)
    x_2 = np.squeeze(x_2)

    dis = 0
    for i in range(len(x_1)):
        dis += (x_1[i]-x_2[i])*(x_1[i]-x_2[i])

    return dis ** 0.5


def find_sub_max(arr, n=2):
    '''

    :param arr: 一维数组
    :param n: n
    :return: 返回数组中第n大的数
    '''
    for i in range(n-1):
        arr_ = arr
        arr_[np.argmax(arr_)] = np.min(arr)
        arr = arr_
    return np.argmax(arr)

def knn_inflo(X, k=4, n=2):
    '''

    :param X: 二维数组
    :param k: k近邻个数
    :param n: 异常点个数 >=1
    :return: INFLO得分最高的n个点
    '''
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    ISpace = []
    ISpace_2 = []
    k_dis = []
    INFLO = []
    for i in range(len(X)):
        ISpace.append([])
        ISpace_2.append([])
        for indice in indices[i][1:]:
            ISpace[i].append(indice)
        k_dis.append(distances[i][-1])

    print(k_dis)

    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            if i in ISpace[j] and j in ISpace[i]:
                ISpace_2[i].append(j)

    print(ISpace)

    for i in range(len(X)):
        tmp = 0
        if len(ISpace_2[i]) == 0:
            tmp = 100000000
        else:
            for item in ISpace_2[i]:
                tmp += 1/k_dis[item]
            tmp /= len(ISpace_2[i])
            tmp *= k_dis[i]
        INFLO.append(tmp)
    #INFLO = np.array(INFLO)
    print(INFLO)
    index = []
    index.append(np.argmax(INFLO))

    for i in range(2, n+1):
        index.append( find_sub_max(INFLO, n=i))
    #index_1, index_2 = arg_sort.index(0), arg_sort.index(1)
    return index










if __name__ == '__main__':

    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0]*34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]
    time_series = df['扬州市秦邮特种金属材料有限公司']
    #values = time_series.values
    #values[30] = (values[29]+values[31])/2
    #values[1] = (values[0]+values[2])/2
    #time_series = Series(values, index=time_series.index)
    # plt.figure(figsize=(6, 6))
    # time_series.plot()
    # plt.title("汽车行业")
    # plt.savefig('./industry_trend/vehicle.jpg')
    values = time_series.values
    x = values[:-1]
    y = values[1:]

    c = np.c_[x, y]
    print(c)
    #anomaly_index = []
    index = knn_inflo(c, n=2)
    #anomaly_index.append(c[index[0]])
    #anomaly_index.append(c[index[1]])
    anomaly_points = np.array([c[i] for i in index])
    normal_points = np.array([c[i] for i in range(len(c)) if i not in index])
    #print(anomaly_index)
    plt.figure(figsize=(6, 6))
    plt.scatter(normal_points[:, 0], normal_points[:, 1], marker='<', edgecolors='b')
    plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], marker='>', edgecolors='r')
    plt.savefig('./anomaly_detection/qytz.jpg')
    plt.show()