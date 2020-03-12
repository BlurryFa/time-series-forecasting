from arch import arch_model
from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE
import numpy as np
import matplotlib.pyplot as plt



def train_test_split(time_series):
    train = time_series[:-2]
    test = time_series[-3:]
    return train, test


if __name__ == '__main__':
    df = data_reading()
    time_series = df['扬州市秦邮特种金属材料有限公司']
    train, test = train_test_split(time_series)
    model = arch_model(train, mean='Zero', vol='ARCH', p=4)
    # 拟合自回归条件异方差模型
    model_fit = model.fit()
    y_pre = model_fit.forecast(horizon=3)



