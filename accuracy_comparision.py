import seaborn as sns
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib ##负号问题
matplotlib.rcParams['axes.unicode_minus']=False

from pylab import mpl##中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']


method_names = ['Ada Boost ', 'FCNN', 'ARIMA', 'CNN', 'Kalman', 'SES', 'Holt Winters', 'Stack LSTM',
                'Bi LSTM', 'Conv1 LSTM', 'Conv2 LSTM', 'Random Forest', 'Recursive LS', 'SVR',
                'Theta', 'XG Boost']

elec_sMAPE = [0.044390349, 0.019685383, 0.015925455, 0.040263006, 0.048443761, 0.03809807, 0.035432057,
              0.085580773, 0.043741685, 0.054033028, 0.04309322, 0.050050323, 0.028963252, 0.020445457,
              0.022481553, 0.036671868]
elec_MSE = [0.003841244, 0.000763034, 0.00026552, 0.003352526, 0.004289984, 0.001656159, 0.001810024, 0.012973678,
            0.003451479, 0.004810837, 0.00302832, 0.005149571, 0.001854959, 0.000874188, 0.001164533, 0.005306419]

elec_df_sMAPE = [0.030645027, 0.007617769, 0.01347455, 0.011130805, 0.014299003, 0.027098079, 0.030432057, 0.005089045,
                 0.008587752, 0.007015729, 0.002995361, 0.01341614, 0.021282586, 0.007075027, 0.006242759, 0.02670968]

elec_df_MSE = [0.000938342, 6.55878E-05, 0.000239449, 0.000160742, 0.000329157, 0.000720616, 0.000910024, 2.57942E-05,
               8.92742E-05, 9.83625E-05, 1.29356E-05, 0.000349271, 0.000454908, 7.75039E-05, 3.92022E-05, 0.001326593]

text_sMAPE = [0.084954229, 0.186415369, 0.133751514, 0.118518382, 0.227457954, 0.136399803, 0.152398053, 0.110942019,
              0.137569672, 0.177976477, 0.180483665, 0.111456788, 0.201753125, 0.172419986, 0.183253037, 0.121067388]

text_MSE = [0.01738335, 0.08401713, 0.035398548, 0.03052168, 0.16059013, 0.042066257, 0.052859938, 0.018118654,
            0.027959276, 0.053142604, 0.064418095, 0.027789925, 0.12833074, 0.044549142, 0.067679643, 0.049922898]

text_df_sMAPE = [0.082597835, 0.085386444, 0.124670512, 0.03131106, 0.114147002, 0.086399803, 0.122398053, 0.077209835,
                 0.075520197, 0.040486321, 0.058622659, 0.083120745, 0.147128302, 0.100729221, 0.103420524, 0.070933521]

text_df_MSE = [0.009807453, 0.012034004, 0.034226619, 0.001499515, 0.019157513, 0.014066257, 0.031359938, 0.007892526,
               0.006502641, 0.002903689, 0.004516263, 0.006933228, 0.024996537, 0.01059213, 0.013969792, 0.011392484]

manu_sMAPE = [0.048422559, 0.175128038, 0.056223906, 0.160269879, 0.057640255, 0.022321652, 0.016832054, 0.126950923,
              0.116929892, 0.129079181, 0.125689961, 0.080965374, 0.14059397, 0.137414933, 0.121362923, 0.067981136]

manu_MSE = [0.010314267, 0.069675636, 0.004328714, 0.070753576, 0.00414619, 0.000665477, 0.000359108, 0.084728441,
            0.080242359, 0.08662083, 0.084969335, 0.017409165, 0.038866195, 0.084739106, 0.086326115, 0.012043149]

manu_df_sMAPE = [0.041075366, 0.083139262, 0.05870605, 0.0511753, 0.043210655, 0.025321652, 0.017832054, 0.027286191,
                 0.013978318, 0.045146253, 0.015759954, 0.02253059, 0.086733439, 0.020918019, 0.061958883, 0.033121878]

manu_df_MSE = [0.003252698, 0.006857602, 0.004428714, 0.002761639, 0.002804211, 0.001065477, 0.000399108, 0.001002161,
               0.000259062, 0.002503557, 0.000461507, 0.000783771, 0.009320925, 0.000713681, 0.004105974, 0.002345749]

cons_sMAPE = [0.035055336, 0.034412139, 0.060401657, 0.057304189, 0.082807411, 0.058214741, 0.05704901, 0.098141864,
              0.064353858, 0.087074798, 0.085270087, 0.055537543, 0.042425707, 0.053258875, 0.056284058, 0.024550411]

cons_MSE = [0.003771491, 0.003073033, 0.00393197, 0.006052731, 0.009249154, 0.00472066, 0.004371011, 0.023434603,
            0.011257705, 0.015094519, 0.017147023, 0.007551408, 0.005560621, 0.004792099, 0.00718212, 0.001264018]

cons_df_sMAPE = [0.027280771, 0.01494295, 0.052904266, 0.01466504, 0.053034236, 0.068234741, 0.05804901, 0.023857141,
                 0.021216646, 0.032987516, 0.0114475, 0.015250184, 0.030879562, 0.02748947, 0.025481538, 0.011814325]

cons_df_MSE = [0.000943689, 0.000332217, 0.00293197, 0.000331446, 0.003026835, 0.01472066, 0.004871011, 0.000818975,
               0.000765686, 0.001945287, 0.000249582, 0.000236394, 0.001418281, 0.000861503, 0.000672561, 0.000223691]

oil_sMAPE = [0.067352109, 0.088401301, 0.090957571, 0.07905797, 0.100248049, 0.075101074, 0.079178937, 0.106832963,
             0.095330718, 0.094551304, 0.094766711, 0.077254695, 0.078094017, 0.081795032, 0.07296834, 0.075000331]

oil_MSE = [0.010210169, 0.019667374, 0.012315932, 0.009445579, 0.018259939, 0.006429104, 0.007419804, 0.024593222,
           0.01474666, 0.017088414, 0.016232883, 0.012734988, 0.020955806, 0.014347431, 0.008939751, 0.011665212]

oil_df_sMAPE = [0.048441508, 0.049700148, 0.060982735, 0.04242416, 0.049219773, 0.072101074, 0.069178937, 0.044365273,
                0.04490816, 0.048843444, 0.045501235, 0.05426606, 0.05728593, 0.051914111, 0.092789749, 0.027663121]

oil_df_MSE = [0.002338367, 0.002615937, 0.000478159, 0.002038159, 0.003987127, 0.005406104, 0.004069804, 0.001962563,
              0.002011103, 0.002377414, 0.002107094, 0.002955273, 0.004516072, 0.003061182, 0.008989177, 0.00116169]

vehi_sMAPE = [0.033193759, 0.027799623, 0.056536218, 0.14294066, 0.040697818, 0.055797413, 0.06093912, 0.118709032,
              0.127191766, 0.072783395, 0.102392972, 0.047116848, 0.074552728, 0.040940341, 0.07702984, 0.041249076]

vehi_MSE = [0.004701792, 0.002511044, 0.004173924, 0.038098102, 0.001827842, 0.003659351, 0.004127027, 0.021910715,
            0.023612589, 0.009103421, 0.017848518, 0.005155627, 0.015289567, 0.002900742, 0.010748966, 0.005352128]

vehi_df_sMAPE = [0.036428867, 0.034400548, 0.052536218, 0.080067084, 0.055142243, 0.049797413, 0.05893912, 0.047094179,
                 0.048636374, 0.016404766, 0.015381722, 0.016885006, 0.057085173, 0.021607454, 0.064867188, 0.024239499]

vehi_df_MSE = [0.001607279, 0.001338549, 0.003173924, 0.00818913, 0.003678443, 0.002809351, 0.003977027, 0.002127271,
               0.004144445, 0.000284661, 0.00016351, 0.000551046, 0.00528304, 0.000528795, 0.00473381, 0.001229223]


all_sMAPE = [elec_sMAPE, text_sMAPE, manu_sMAPE, cons_sMAPE, oil_sMAPE, vehi_sMAPE]
all_sMAPE = np.array(all_sMAPE)
all_MSE = [elec_MSE, text_MSE, manu_MSE, cons_MSE, oil_MSE, vehi_MSE]
all_MSE = np.array(all_MSE)
all_df_sMAPE = [elec_df_sMAPE, text_df_sMAPE, manu_df_sMAPE, cons_df_sMAPE, oil_df_sMAPE, vehi_df_sMAPE]
all_df_sMAPE = np.array(all_df_sMAPE)
all_df_MSE = [elec_df_MSE, text_df_MSE, manu_df_MSE, cons_df_MSE, oil_df_MSE, vehi_df_MSE]
all_df_MSE = np.array(all_df_MSE)

if __name__ == '__main__':
    # plt.figure(figsize=(8, 8))
    # sns.barplot(x=text_df_sMAPE, y=method_names, orient='h') #画水平的柱状图
    # plt.title('纺织产业sMAPE(差分)')# for Industry(差分)
    # plt.savefig('./model_compare/textsMAPEdf.jpg')
    # plt.show()
    xg_sMAPE = Series(all_sMAPE[:, 15])
    theta_sMAPE = Series(all_sMAPE[:, 14])
    svr_sMAPE = Series(all_sMAPE[:, 13])
    rls_sMAPE = Series(all_sMAPE[:, 12])
    rf_sMAPE = Series(all_sMAPE[:, 11])
    Conv2_LSTM_sMAPE = Series(all_sMAPE[:, 10])
    Conv1_LSTM_sMAPE = Series(all_sMAPE[:, 9])
    Bi_LSTM_sMAPE = Series(all_sMAPE[:, 8])
    Stack_LSTM_sMAPE = Series(all_sMAPE[:, 7])
    Holt_Winters_sMAPE = Series(all_sMAPE[:, 6])
    SES_sMAPE = Series(all_sMAPE[:, 5])
    Kalman_sMAPE = Series(all_sMAPE[:, 4])
    CNN_sMAPE = Series(all_sMAPE[:, 3])
    ARIMA_sMAPE = Series(all_sMAPE[:, 2])
    FCNN_sMAPE = Series(all_sMAPE[:, 1])
    ada_boost_sMAPE = Series(all_sMAPE[:, 0])

    plt.figure(figsize=(6, 6))
    xg_sMAPE.plot(label='XG Boost', legend=True, color='k')
    theta_sMAPE.plot(label='Theta', legend=True, color='r')
    svr_sMAPE.plot(label='SVR', legend=True, color='peru')
    rls_sMAPE.plot(label='Recursive LS', legend=True, color='orange')
    rf_sMAPE.plot(label='Random Forest', legend=True, color='gold')
    Conv2_LSTM_sMAPE.plot(label='Conv2 LSTM', legend=True, color='y')
    Conv1_LSTM_sMAPE.plot(label='Conv1 LSTM', legend=True, color='g')
    Bi_LSTM_sMAPE.plot(label='Bi LSTM', legend=True, color='greenyellow')
    Stack_LSTM_sMAPE.plot(label='Stack LSTM', legend=True, color='c')
    Holt_Winters_sMAPE.plot(label='Holt Winters', legend=True, color='b')
    SES_sMAPE.plot(label='SES', legend=True, color='m')
    Kalman_sMAPE.plot(label='Kalman', legend=True, color='slategrey')
    CNN_sMAPE.plot(label='CNN', legend=True, color='blueviolet')
    ARIMA_sMAPE.plot(label='ARIMA', legend=True, color='fuchsia')
    FCNN_sMAPE.plot(label='FCNN', legend=True, color='silver')
    ada_boost_sMAPE.plot(label='Ada Boost', legend=True, color='deepskyblue')
    plt.savefig('./modelfigure/sMAPE.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    xg_sMAPE.plot(kind='kde', label='XG Boost', legend=True, color='k')
    theta_sMAPE.plot(kind='kde', label='Theta', legend=True, color='r')
    svr_sMAPE.plot(kind='kde', label='SVR', legend=True, color='peru')
    rls_sMAPE.plot(kind='kde', label='Recursive LS', legend=True, color='orange')
    rf_sMAPE.plot(kind='kde', label='Random Forest', legend=True, color='gold')
    Conv2_LSTM_sMAPE.plot(kind='kde', label='Conv2 LSTM', legend=True, color='y')
    Conv1_LSTM_sMAPE.plot(kind='kde', label='Conv1 LSTM', legend=True, color='g')
    Bi_LSTM_sMAPE.plot(kind='kde', label='Bi LSTM', legend=True, color='greenyellow')
    Stack_LSTM_sMAPE.plot(kind='kde', label='Stack LSTM', legend=True, color='c')
    Holt_Winters_sMAPE.plot(kind='kde', label='Holt Winters', legend=True, color='b')
    SES_sMAPE.plot(kind='kde', label='SES', legend=True, color='m')
    Kalman_sMAPE.plot(kind='kde', label='Kalman', legend=True, color='slategrey')
    CNN_sMAPE.plot(kind='kde', label='CNN', legend=True, color='blueviolet')
    ARIMA_sMAPE.plot(kind='kde', label='ARIMA', legend=True, color='fuchsia')
    FCNN_sMAPE.plot(kind='kde', label='FCNN', legend=True, color='silver')
    ada_boost_sMAPE.plot(kind='kde', label='Ada Boost', legend=True, color='deepskyblue')
    plt.xlim((0, 0.2))
    plt.savefig('./modelfigure/sMAPEkde.jpg')
    plt.show()

    xg_df_sMAPE = Series(all_df_sMAPE[:, 15])
    theta_df_sMAPE = Series(all_df_sMAPE[:, 14])
    svr_df_sMAPE = Series(all_df_sMAPE[:, 13])
    rls_df_sMAPE = Series(all_df_sMAPE[:, 12])
    rf_df_sMAPE = Series(all_df_sMAPE[:, 11])
    Conv2_LSTM_df_sMAPE = Series(all_df_sMAPE[:, 10])
    Conv1_LSTM_df_sMAPE = Series(all_df_sMAPE[:, 9])
    Bi_LSTM_df_sMAPE = Series(all_df_sMAPE[:, 8])
    Stack_LSTM_df_sMAPE = Series(all_df_sMAPE[:, 7])
    Holt_Winters_df_sMAPE = Series(all_df_sMAPE[:, 6])
    SES_df_sMAPE = Series(all_df_sMAPE[:, 5])
    Kalman_df_sMAPE = Series(all_df_sMAPE[:, 4])
    CNN_df_sMAPE = Series(all_df_sMAPE[:, 3])
    ARIMA_df_sMAPE = Series(all_df_sMAPE[:, 2])
    FCNN_df_sMAPE = Series(all_df_sMAPE[:, 1])
    ada_boost_df_sMAPE = Series(all_df_sMAPE[:, 0])

    plt.figure(figsize=(6, 6))
    xg_df_sMAPE.plot(label='XG Boost', legend=True, color='k')
    theta_df_sMAPE.plot(label='Theta', legend=True, color='r')
    svr_df_sMAPE.plot(label='SVR', legend=True, color='peru')
    rls_df_sMAPE.plot(label='Recursive LS', legend=True, color='orange')
    rf_df_sMAPE.plot(label='Random Forest', legend=True, color='gold')
    Conv2_LSTM_df_sMAPE.plot(label='Conv2 LSTM', legend=True, color='y')
    Conv1_LSTM_df_sMAPE.plot(label='Conv1 LSTM', legend=True, color='g')
    Bi_LSTM_df_sMAPE.plot(label='Bi LSTM', legend=True, color='greenyellow')
    Stack_LSTM_df_sMAPE.plot(label='Stack LSTM', legend=True, color='c')
    Holt_Winters_df_sMAPE.plot(label='Holt Winters', legend=True, color='b')
    SES_df_sMAPE.plot(label='SES', legend=True, color='m')
    Kalman_df_sMAPE.plot(label='Kalman', legend=True, color='slategrey')
    CNN_df_sMAPE.plot(label='CNN', legend=True, color='blueviolet')
    ARIMA_df_sMAPE.plot(label='ARIMA', legend=True, color='fuchsia')
    FCNN_df_sMAPE.plot(label='FCNN', legend=True, color='silver')
    ada_boost_df_sMAPE.plot(label='Ada Boost', legend=True, color='deepskyblue')
    plt.savefig('./modelfigure/sMAPEdf.jpg')
    plt.show()

    plt.figure(figsize=(6, 6))
    xg_df_sMAPE.plot(kind='kde', label='XG Boost', legend=True, color='k')
    theta_df_sMAPE.plot(kind='kde', label='Theta', legend=True, color='r')
    svr_df_sMAPE.plot(kind='kde', label='SVR', legend=True, color='peru')
    rls_df_sMAPE.plot(kind='kde', label='Recursive LS', legend=True, color='orange')
    rf_df_sMAPE.plot(kind='kde', label='Random Forest', legend=True, color='gold')
    Conv2_LSTM_df_sMAPE.plot(kind='kde', label='Conv2 LSTM', legend=True, color='y')
    Conv1_LSTM_df_sMAPE.plot(kind='kde', label='Conv1 LSTM', legend=True, color='g')
    Bi_LSTM_df_sMAPE.plot(kind='kde', label='Bi LSTM', legend=True, color='greenyellow')
    Stack_LSTM_df_sMAPE.plot(kind='kde', label='Stack LSTM', legend=True, color='c')
    Holt_Winters_df_sMAPE.plot(kind='kde', label='Holt Winters', legend=True, color='b')
    SES_df_sMAPE.plot(kind='kde', label='SES', legend=True, color='m')
    Kalman_df_sMAPE.plot(kind='kde', label='Kalman', legend=True, color='slategrey')
    CNN_df_sMAPE.plot(kind='kde', label='CNN', legend=True, color='blueviolet')
    ARIMA_df_sMAPE.plot(kind='kde', label='ARIMA', legend=True, color='fuchsia')
    FCNN_df_sMAPE.plot(kind='kde', label='FCNN', legend=True, color='silver')
    ada_boost_df_sMAPE.plot(kind='kde', label='Ada Boost', legend=True, color='deepskyblue')
    plt.xlim((0, 0.2))
    plt.savefig('./modelfigure/sMAPEdfkde.jpg')
    plt.show()
