from data_reading import data_reading
from industrial_company import *
from pandas import DataFrame, Series
import numpy as np
from data_preprocessing import *
from season_decomposition import de_seasonality
import matplotlib.pyplot as plt
import pickle as pk

import matplotlib##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == '__main__':
    #df = data_reading()
    #df, se = de_seasonality(df)
    with open('./residual_pickle/rls_oil.pkl', 'rb') as f:
        rls_oil = pk.load(f)
    with open('./residual_pickle/svr_oil.pkl', 'rb') as f:
        svr_oil = pk.load(f)
    with open('./residual_pickle/es_oil.pkl', 'rb') as f:
        es_oil = pk.load(f)
    with open('./residual_pickle/holt_oil.pkl', 'rb') as f:
        holt_oil = pk.load(f)
    with open('./residual_pickle/dkf_oil.pkl', 'rb') as f:
        dkf_oil = pk.load(f)
    with open('./residual_pickle/adaboost_oil.pkl', 'rb') as f:
        adaboost_oil = pk.load(f)
    with open('./residual_pickle/arima_oil_df.pkl', 'rb') as f:
        arima_oil = pk.load(f)
    with open('./residual_pickle/bilstm_oil.pkl', 'rb') as f:
        bilstm_oil = pk.load(f)
    with open('./residual_pickle/cnn_oil.pkl', 'rb') as f:
        cnn_oil = pk.load(f)
    with open('./residual_pickle/con1lstm_oil.pkl', 'rb') as f:
        con1lstm_oil = pk.load(f)
    with open('./residual_pickle/con2lstm_oil.pkl', 'rb') as f:
        con2lstm_oil = pk.load(f)
    with open('./residual_pickle/fcnn_oil.pkl', 'rb') as f:
        fcnn_oil = pk.load(f)
    with open('./residual_pickle/randomforest_oil.pkl', 'rb') as f:
        randomforest_oil = pk.load(f)
    with open('./residual_pickle/stacklstm_oil.pkl', 'rb') as f:
        stacklstm_oil = pk.load(f)
    with open('./residual_pickle/theta_oil.pkl', 'rb') as f:
        theta_oil = pk.load(f)
    with open('./residual_pickle/xgboost_oil.pkl', 'rb') as f:
        xgboost_oil = pk.load(f)

    # plt.figure(figsize=(6, 6))
    # xgboost_oil.plot(kind='kde', label='XG Boost', legend=True)
    # adaboost_oil.plot(kind='kde', label='Ada Boost', legend=True)
    # plt.grid(which='both')
    # plt.title('汽车产业')
    # plt.savefig('./resiplot/residualoiladaxg.jpg')
    plt.figure(figsize=(6, 6))
    svr_oil.plot(kind='kde', label='SVR', legend=True, color='k')
    arima_oil.plot(kind='kde', label='ARIMA', legend=True, color='r')
    rls_oil.plot(kind='kde', label='Recursive LS', legend=True, color='peru')
    dkf_oil.plot(kind='kde', label='Discrete Kalman', legend=True, color='orange')
    es_oil.plot(kind='kde', label='SES', legend=True, color='gold')
    holt_oil.plot(kind='kde', label='Holt Winters', legend=True, color='y')
    randomforest_oil.plot(kind='kde', label='Random Forest', legend=True, color='g')
    cnn_oil.plot(kind='kde', label='CNN', legend=True, color='greenyellow')
    theta_oil.plot(kind='kde', label='Theta', legend=True, color='c')
    fcnn_oil.plot(kind='kde', label='FCNN', legend=True, color='b')
    bilstm_oil.plot(kind='kde', label='Bidirectional LSTM', legend=True, color='m')
    stacklstm_oil.plot(kind='kde', label='Stack LSTM', legend=True, color='slategrey')
    con1lstm_oil.plot(kind='kde', label='Conv1 LSTM', legend=True, color='silver')
    con2lstm_oil.plot(kind='kde', label='Conv2 LSTM', legend=True, color='deepskyblue')
    plt.grid(which='both')
    plt.title('石化产业')
    plt.savefig('./resiplot/residualoil.jpg')
    # plt.figure(figsize=(6, 6))
    # svr_oil.plot(kind='kde', label='SVR', legend=True)
    # arima_oil.plot(kind='kde', label='ARIMA', legend=True)
    # rls_oil.plot(kind='kde', label='Recursive LS', legend=True)
    # dkf_oil.plot(kind='kde', label='Discrete Kalman', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/svroildf.jpg')
    # plt.figure(figsize=(6, 6))
    # es_oil.plot(kind='kde', label='SES', legend=True)
    # holt_oil.plot(kind='kde', label='Holt Winters', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/sesoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # randomforest_oil.plot(kind='kde', label='Random Forest', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/rfoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # cnn_oil.plot(kind='kde', label='CNN', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/cnnoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # theta_oil.plot(kind='kde', label='Theta', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/thetaoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # fcnn_oil.plot(kind='kde', label='FCNN', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/fcnnoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # bilstm_oil.plot(kind='kde', label='Bidirectional LSTM', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/bilstmoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # adaboost_oil.plot(kind='kde', label='Ada Boost', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/adaoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # stacklstm_oil.plot(kind='kde', label='Stack LSTM', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/stacklstmoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # con1lstm_oil.plot(kind='kde', label='Conv1 LSTM', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/c1lstmoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # xgboost_oil.plot(kind='kde', label='XG Boost', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/xgoildf.jpg')
    # plt.figure(figsize=(6, 6))
    # con2lstm_oil.plot(kind='kde', label='Conv2 LSTM', legend=True)
    # plt.grid(which='both')
    # plt.title('石化产业')
    # plt.savefig('./resiplot/c2lstmoildf.jpg')










    #plt.show()




