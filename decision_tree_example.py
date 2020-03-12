from adaboost_fitting import AdaBoostRegressorTS
from randomforest_fitting import RandomForestTS
from xgboost_fitting import XGboostTS
from data_reading import data_reading
from season_decomposition import de_seasonality
from industrial_company import *
import matplotlib.pyplot as plt
from season_decomposition import de_seasonality
import numpy as np
import pickle as pk
import matplotlib     ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False
from pandas import Series



if __name__ == '__main__':
    df = data_reading()
    df, se = de_seasonality(df)
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0] * 34, index=df.index)
    for enterprise in construction:
        time_series = time_series + df[enterprise]
    time_series = time_series.diff(1)
    time_series = time_series.dropna()
    time_series = df['扬州市秦邮特种金属材料有限公司']

    # bias = Series([2000]*len(time_series), index=time_series.index)
    # time_series = time_series + bias

    adaboost = AdaBoostRegressorTS()
    adaboost.initialize()
    randomforest = RandomForestTS()
    randomforest.initialize()
    xgboost = XGboostTS()
    xgboost.initialize()

    train_X, train_y = adaboost.split_sequence(time_series)

    adaboost.train(train_X, train_y)
    randomforest.train(train_X, train_y)
    xgboost.train(train_X, train_y)

    pre_ada = adaboost.predict(train_X)
    pre_rf = randomforest.predict(train_X)
    pre_xg = xgboost.predict(train_X)

    index = time_series.index[4:]

    pre_ada = Series(pre_ada, index=index)
    pre_rf = Series(pre_rf, index=index)
    pre_xg = Series(pre_xg, index=index)
    # test = Series(test_values, index=index)
    plt.figure(figsize=(6, 6))
    plt.subplot(311)
    pre_ada.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_ada = time_series - pre_ada
    # with open('./residual_pickle/adaboost_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_ada, f)
    plt.title('Ada Boost')
    plt.grid(which='both')

    plt.subplot(312)
    pre_rf.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_rf = time_series - pre_rf
    # with open('./residual_pickle/randomforest_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_rf, f)
    plt.title('Random Forest')
    plt.grid(which='both')

    plt.subplot(313)
    pre_xg.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_xg = time_series - pre_xg
    # with open('./residual_pickle/xgboost_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_xg, f)
    plt.title('XG Boost')
    plt.grid(which='both')

    plt.savefig('./pre_plot/ensemble.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual_ada.plot(label='residual for Ada Boost', legend=True)
    residual_rf.plot(label='residual for Random Forest', legend=True)
    residual_xg.plot(label='residual for Xg Boost', legend=True)
    plt.grid(which='both')

    plt.subplot(212)
    residual_ada.plot(kind='kde', label='residual density for Ada Boost', legend=True)
    residual_rf.plot(kind='kde', label='residual density for Random Forest', legend=True)
    residual_xg.plot(kind='kde', label='residual density for Xg Boost', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/ensembler.jpg')
    plt.show()



