import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# **************** differential
def differential(time_series, step=1):
    time_series = time_series.diff(step)
    return time_series.dropna()


# **************** log transform
def log_transform(time_series):
     values = np.log(time_series.values)
     time_series_log = Series(values, index=time_series.index)
     return time_series_log


# **************** Box Cox power transform
def Box_Cox_power_transform(time_series):
    values = time_series.values
    values, para_lambda = stats.boxcox(values)
    time_series_bct = Series(values, index=time_series.index)

    return time_series_bct, para_lambda


# **************** 检测是否有趋势成分存在的
def Cox_Stuart_test(time_series):
    '''
    :param time_series: pandas.series()
    :return: 是否稳定 bool
    '''
    values = list(time_series.values)
    n = len(values)
    S_plus = 0
    S_minus = 0

    if n % 2 == 0:
        c = n/2
        n_ = c
        for i in range(c):
            if values[i] - values[i+c] > 0:
                S_plus += 1
            elif values[i] - values[i+c] < 0:
                S_minus += 1
    else:
        c = (n+1)/2
        n_ = c-1
        for i in range(c-1):
            if values[i] - values[i+c] > 0:
                S_plus += 1
            elif values[i] - values[i+c] < 0:
                S_minus += 1

    k = min(S_plus, S_minus)

    a = 0.002
    #   计算p值
    p = 0
    #   计算排列组合
    for i in range(1, k+1):
        temp = 1
        for j in range(i):
            temp *= (n_-j)
            temp /= (j+1)
        p += temp

    p /= 2**(n_-1)

    if p > a:
        return False   ##无趋势
    else:
        return True    ##有趋势


# **************** 稳定性校验
def is_stable(time_series):
    values = time_series.values
    t = adfuller(values)

    if t[1] < 0.05: #t[1]是检验的p值，小于0.05显著性，拒绝原假设
        return True
    return False


# **************** 判断是否是白噪声
def is_white_noise(time_series):
    values = time_series.values
    p = acorr_ljungbox(values, lags=1)[1]

    if p < 0.05:
        return False
    return True
# **************** 标准化数据


def standard_scale(time_series):
    values = time_series.values
    max_value = max(values)
    min_value = min(values)
    for i in range(len(values)):
        values[i] = (values[i]-min_value)/(max_value-min_value)

    time_series_standard = Series(values, index=time_series.index)

    return time_series_standard, max_value, min_value











