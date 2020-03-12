from industrial_company import *
from DiscreteKalmanFilter import DiscreteKalmanFilter
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE, MSE
from season_decomposition import de_seasonality
import pickle as pk
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']



def split_sequence(time_series, n_steps):
    X, y, z = list(), list(), list()
    values = time_series.values
    for i in range(len(values)):
        end_ix = i + n_steps

        if end_ix > len(values) - 1:
            break

        seq_x, seq_y = values[i:end_ix], values[i:end_ix]
        X.append(seq_x)
        y.append(seq_y)
        z.append(values[end_ix])

    return np.array(X), np.array(y), np.array(z)


def train_test_split(time_series):
    train = time_series[:-1]
    test = time_series[-6:]
    return train, test


if __name__ == "__main__":
    df = data_reading()
    df, se = de_seasonality(df)
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0] * 34, index=df.index)
    for enterprise in oil:
        time_series = time_series + df[enterprise]
    time_series_df = time_series.diff(1)
    time_series_df = time_series.dropna()
    time_series = df['扬州市秦邮特种金属材料有限公司']


    train, test = train_test_split(time_series)
    train_X, train_y, _ = split_sequence(train, 4)
    test_X, test_x, test_y = split_sequence(test, 4)

    dkf = DiscreteKalmanFilter()

    y_pre = []

    #dkf.init(train_X, train_y)
    #
    # for x, z in zip(test_X, test_x):
    #     y_pre.append(dkf.update(z))
    #y_pre = Series(y_pre)
    #test_y = Series(test_y)

    # index = time_series.index[-2:]
    # values = time_series.values
    # tmp_1 = values[-3] + y_pre[0]
    # tmp_2 = values[-2] + y_pre[1]
    # y_pre = Series([tmp_1, tmp_2], index=index)
    #
    # tmp_1 = values[-3] + test_y[0]
    # tmp_2 = values[-2] + test_y[1]
    # test_y = Series([tmp_1, tmp_2], index=index)
    # print(sMAPE(y_pre, test_y))
    # print(MSE(y_pre, test_y))



    _, z_plot, _ =  split_sequence(time_series, 4)
    dkf.init(train_X, train_y)
    z_plot = z_plot[1:]

    dkf.x = z_plot[0].reshape((1, 4))
    y_pre = []
    for z in z_plot:
        y_pre.append(dkf.update(z))
    y_pre = y_pre[:-15]
    dkf.init(train_X, train_y)
    for z in z_plot[-15:]:
        y_pre.append(dkf.update(z))
    #
    index_plot = time_series.index[5:]
    pre_plot = Series(y_pre, index=index_plot)
    plt.figure(figsize=(6, 6))

    pre_plot.plot(color='green', label='Predicts', legend=True)  # , label='Predicts'
    time_series.plot(color='blue', label='Original', legend=True)  # , label='Original'
    residual = time_series - pre_plot
    # with open('./residual_pickle/dkf_oil_df.pkl', 'wb') as f:
    #     pk.dump(residual, f)
    plt.title('discrete kalman')
    plt.grid(which='both')
    plt.savefig('./pre_plot/dkf.jpg')
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual.plot(label='residual for discrete kalman', legend=True)
    plt.grid(which='both')
    plt.subplot(212)
    residual.plot(kind='kde', label='residual kernel density for discrete kalman', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/dkfr.jpg')
    plt.show()




