import numpy as np
from pandas import Series
from data_reading import data_reading
import matplotlib.pyplot as plt

def moving_average(values, kernel):
    '''
    :param values: 一维数组
    :param kernel:  移动平均窗口权值
    :return: 移动平均后的数组
    '''
    n_kernel = int(len(kernel)/2)
    n = len(values)
    values_ma = [0] * n
    if n_kernel%2 == 0:
        for i in range(n):

            if i < n_kernel - 1 or i >= n - n_kernel:
                values_ma[i] = 0
            else:
                values_ma[i] = 0
                for j in range(-n_kernel + 1, n_kernel + 1):
                    values_ma[i] += values[i + j] * kernel[j + n_kernel - 1]
        values_ma = np.array(values_ma)
    else:
        values_ma = [0] * n
        for i in range(n):

            if i < n_kernel or i >= n - n_kernel:
                values_ma[i] = 0
            else:
                values_ma[i] = 0
                for j in range(-n_kernel, n_kernel + 1):
                    values_ma[i] += values[i + j] * kernel[j + n_kernel]
        values_ma = np.array(values_ma)

    return values_ma


def x_11(time_series):
    '''

    :param time_series: pandas.Series()
    :return: 季节性序列Series() ，剔除季节性的原始序列 Series()
    '''

    values = np.array(time_series.values)
    M_212 = np.array([1,2,2,2,2,2,2,2,2,2,2,1])/24
    M_33 = np.array([1,2,3,2,1])/9
    M_315 = np.array([1,2,3,3,3,2,1])/15
    Henderson_13 = np.array([-0.019, -0.028, 0.0, 0.066, 0.147, 0.214, 0.240, 0.214, 0.147, 0.066, 0.0, -0.028, -0.019])

    T_1 = moving_average(values, M_212)
    S_1 = values - T_1

    S_1a = moving_average(S_1, M_33)
    S_1a = S_1a - moving_average(S_1a, M_212)

    S_at = values - S_1a

    T_2 = moving_average(S_at, Henderson_13)
    S_2 = values - T_2
    S_2a = moving_average(S_2, M_315)
    S_2a = S_2a - moving_average(S_2a, M_212)

    return Series(S_2a, index=time_series.index), Series(values-S_2a, index=time_series.index)

if __name__ == '__main__':
    df = data_reading()
    fonsize = {'fontsize': 'xx-large',
 'fontweight': 'black',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}
    ts = df['扬州市秦邮特种金属材料有限公司']
    #print(rcParams['axes.titlesize'])
    df, se = x_11(ts)
    df.plot()
    plt.show()
    se.plot()
    plt.show()






