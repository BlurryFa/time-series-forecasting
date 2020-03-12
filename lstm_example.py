from lstm_fitting import Stack_Lstm, BidirectionalLstm, CnnLstm, ConvLstm
from data_reading import data_reading
from data_preprocessing import *
import matplotlib.pyplot as plt
from industrial_company import *
from season_decomposition import de_seasonality
import numpy as np
import pickle as pk
import matplotlib         ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

if __name__ == '__main__':
    stack_lstm = Stack_Lstm()
    stack_lstm.initialize()
    stack_lstm.plot_model()

    bi_lstm = BidirectionalLstm()
    bi_lstm.initialize()
    bi_lstm.plot_model()

    con1_lstm = CnnLstm()
    con1_lstm.initialize()
    con1_lstm.plot_model()

    con2_lstm = ConvLstm()
    con2_lstm.initialize()
    con2_lstm.plot_model()



    df = data_reading()
    df, se = de_seasonality(df)
    #time_series = df['扬州市秦邮特种金属材料有限公司']
    time_series = Series([0] * 34, index=df.index)
    for enterprise in construction:
        time_series = time_series + df[enterprise]
    time_series = time_series.diff(1)
    time_series = time_series.dropna()
    time_series = df['扬州市秦邮特种金属材料有限公司']
    train_X, train_y = stack_lstm.split_sequence(time_series)

    stack_lstm.train(train_X, train_y)
    bi_lstm.train(train_X, train_y)
    con1_lstm.train(train_X, train_y)
    con2_lstm.train(train_X, train_y)

    pre_stack = stack_lstm.predict(train_X)
    pre_bi = bi_lstm.predict(train_X)
    pre_con1 = con1_lstm.predict(train_X)
    pre_con2 = con2_lstm.predict(train_X)

    index = time_series.index[4:]

    pre_stack = Series(pre_stack, index=index)
    pre_bi = Series(pre_bi, index=index)
    pre_con1 = Series(pre_con1, index=index)
    pre_con2 = Series(pre_con2, index=index)

    plt.figure(figsize=(6, 6))
    plt.subplot(221)
    pre_stack.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_stack = time_series - pre_stack
    # with open('./residual_pickle/stacklstm_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_stack, f)
    print(residual_stack)
    plt.title('Stack LSTM')
    plt.grid(which='both')

    plt.subplot(222)
    pre_bi.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_bi = time_series - pre_bi
    # with open('./residual_pickle/bilstm_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_bi, f)
    print(residual_bi)
    plt.title('Bidirectional LSTM')
    plt.grid(which='both')

    plt.subplot(223)
    pre_con1.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_con1 = time_series - pre_con1
    # with open('./residual_pickle/con1lstm_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_con1, f)
    plt.title('Conv1 LSTM')
    plt.grid(which='both')

    plt.subplot(224)
    pre_con2.plot(color='green', label='Predicts', legend=True)
    time_series.plot(color='blue', label='Original', legend=True)
    residual_con2 = time_series - pre_con2
    # with open('./residual_pickle/con2lstm_construction_df.pkl', 'wb') as f:
    #     pk.dump(residual_con2, f)
    plt.title('Conv2 LSTM')
    plt.grid(which='both')


    plt.savefig('./pre_plot/LSTM.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    residual_stack.plot(label='residual for Stack LSTM', legend=True)
    residual_bi.plot(label='residual for Bi-LSTM', legend=True)
    residual_con1.plot(label='residual for Conv1 LSTM', legend=True)
    residual_con2.plot(label='residual for Conv2 LSTM', legend=True)
    plt.grid(which='both')

    plt.subplot(212)
    residual_stack.plot(kind='kde', label='residual density for Stack LSTM', legend=True)
    residual_bi.plot(kind='kde', label='residual density for Bi-LSTM', legend=True)
    residual_con1.plot(kind='kde', label='residual density for Conv1 LSTM', legend=True)
    residual_con2.plot(kind='kde', label='residual density for Conv2 LSTM', legend=True)
    plt.grid(which='both')
    plt.savefig('./pre_plot/LSTMr.jpg')
    plt.show()





