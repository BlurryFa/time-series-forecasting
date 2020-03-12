from data_reading import data_reading
import matplotlib.pyplot as plt
from pandas import Series
import pandas as pd
import seaborn as sns
from data_preprocessing import standard_scale
from industrial_company import *
import numpy as np
from season_decomposition import de_seasonality



import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

industry_list = ['电子', '纺织', '制造', '建材', '石化', '汽车']

if __name__ == '__main__':
    temperature = []
    with open('./temperature.txt') as f:
        for line in f.readlines():
            if float(line[:-1]) < 16:
                temperature.append(16 - float(line[:-1]))
            elif float(line[:-1]) > 18:
                temperature.append(float(line[:-1]) - 18)
            else:
                temperature.append(0)
    print(temperature)
    index = pd.PeriodIndex(start='2016-01', freq='M', periods=34)
    index = index.to_timestamp()
    ts_temperature = Series(temperature, index=index)
    ts_temperature, _, _ = standard_scale(ts_temperature)

    df = data_reading()
    df, se = de_seasonality(df)
    covariances = []
    num_list = df.columns
    #print(df.columns)
    # for corporation in se.columns:
    #     time_series = se[corporation]
    #     time_series, _, _ = standard_scale(time_series)
    #     covariances.append(abs(time_series.cov(ts_temperature)))

    for industry in industries:
        values = np.array([0]*34)
        for company in industry:
            time_series = df[company]
            values = values + np.array(time_series.values)
        time_series = Series(values, time_series.index)
        time_series, _, _ = standard_scale(time_series)
        covariances.append(abs(time_series.cov(ts_temperature)))


    plt.figure(figsize=(6, 6))
    sns.barplot(x=covariances, y=industry_list, orient='h') #画水平的柱状图
    plt.title('行业季节性电量-温差协方差')# for Industry
    plt.savefig('./cov/covarianceindus.jpg')
    plt.show()
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    #values = time_series.values
    #print(ts_temperature.corr(time_series))



