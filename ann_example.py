from pandas import Series
from data_reading import data_reading
from data_preprocessing import *
from data_evaluating import sMAPE
from industrial_company import *
from ann_fitting import AnnTS
from cnn_fitting import CnnTS
import matplotlib.pyplot as plt
from season_decomposition import de_seasonality
import numpy as np
import pickle as pk
import matplotlib         ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

if __name__ == '__main__':
    cnn = CnnTS()
    cnn.initialize()
    ann = AnnTS()
    ann.initialize()

    df = data_reading()
    df, se = de_seasonality(df)
    time_series = Series([0]*34, index=df.index)
    for enterprise in vehicle:
        time_series = time_series + df[enterprise]

    time_series = time_series.diff(1)
    time_series = time_series.dropna()


    time_series = df['扬州市秦邮特种金属材料有限公司']

    train_X, train_y = cnn.split_sequence(time_series)

    ann.train(train_X, train_y)
    cnn.train(train_X, train_y)

    pre_cnn = cnn.predict(train_X)
    pre_ann = ann.predict(train_X)

    index = time_series.index[4:]

    pre_cnn = Series(pre_cnn, index=index)
    pre_ann = Series(pre_ann, index=index)
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    pre_cnn.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_cnn = time_series - pre_cnn
    # with open('./residual_pickle/cnn_vehicle_df.pkl', 'wb') as f:
    #     pk.dump(residual_cnn, f)
    plt.title('CNN')
    plt.grid(which='both')

    plt.subplot(212)
    pre_ann.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_ann = time_series - pre_ann
    # with open('./residual_pickle/fcnn_vehicle_df.pkl', 'wb') as f:
    #     pk.dump(residual_ann, f)
    plt.title('FCNN')
    plt.grid(which='both')


    plt.savefig('./pre_plot/ffnn.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual_cnn.plot(label='residual for CNN', legend=True)
    residual_ann.plot(label='residual for FCNN', legend=True)
    plt.grid(which='both')

    plt.subplot(212)
    residual_cnn.plot(kind='kde', label='residual density for CNN', legend=True)
    residual_ann.plot(kind='kde', label='residual density for FCNN', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/ffnnr.jpg')
    plt.show()