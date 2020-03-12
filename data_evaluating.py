from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series


def mse(pre_time_series, test):
    pre_values = pre_time_series.values
    test_values = test.values
    return mean_squared_error(pre_values, test_values)


def mbe(pre_time_series, test):
    pre_values = pre_time_series.values
    test_values = test.values
    return mean_absolute_error(pre_values, test_values)


def sMAPE(pre_time_series, test):
    pre_values = pre_time_series.values
    test_values = test.values

    k = len(pre_values)
    temp = 0

    for i in range(0, k):
        a = abs(pre_values[i]-test_values[i])/(abs(pre_values[i])+abs(test_values[i]))
        temp += a

    temp = temp * 2 / k
    return temp


def MSE(pre_time_series, test):
    pre_values = pre_time_series.values
    test_values = test.values
    out = 0
    for i in range(len(pre_values)):
        tmp = (pre_values[i] - test_values[i])**2/(test_values[i]**2)
        out += tmp

    out /= len(pre_values)

    return out



def MASE(pre_time_series, test, origin):
    pre_values = pre_time_series.values
    test_values = test.values
    origin_values = origin.values

    k = len(pre_values)
    temp_1 = 0
    temp_2 = 0
    m = 7
    n = len(origin_values)

    for i in range(k):
        a = abs(pre_values[i]-test_values[i])
        temp_1 += a
    temp_1 /= k

    for i in range(m, n):
        temp_2 += abs(origin_values[i]-origin_values[i-m])
    temp_2 /= (n-m)
    return temp_1/temp_2





def residual_plot(pre_time_series, test):
    pre_values = pre_time_series.values
    test_values = test.values
    residuals = [test_values[i] - pre_values[i]  for i in range(len(pre_values))]
    residuals_series = Series(residuals, index=pre_values.index)
    sns.distplot(residuals_series)
    sns.plt.show()









