
import scipy.io as sio
import math
import random
import matplotlib.pyplot as plt
from numpy import *
#import ICA
n_components = 2

def f1(x, period = 4):
    return 0.5*(x-math.floor(x/period)*period)

def create_data():
    #data number
    n = 500
    #data time
    T = [0.1*xi for xi in range(0, n)]
    #source
    S = array([[sin(xi)  for xi in T], [f1(xi) for xi in T]], float32)
    #mix matrix
    A = array([[0.8, 0.2], [-0.3, -0.7]], float32)
    return T, S, dot(A, S)
def whiten(X):
    #zero mean
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, newaxis]
    #whiten
    A = dot(X, X.transpose())
    D , E = linalg.eig(A)
    # D2 = linalg.inv(array([[D[0], 0.0], [0.0, D[1]]], float32))
    # D2[0,0] = sqrt(D2[0,0]); D2[1,1] = sqrt(D2[1,1])
    D2=sqrt(linalg.inv(diag(D)))
    V = dot(D2, E.transpose())
    return dot(V, X), V
def _logcosh(x, fun_args=None, alpha = 1):
    gx = tanh(alpha * x, x);
    g_x = gx ** 2;
    g_x -= 1.;
    g_x *= -alpha
    return gx, g_x.mean(axis=-1)
def do_decorrelation(W):
    #black magic
    s, u = linalg.eigh(dot(W, W.T))
    return dot(dot(u * (1. / sqrt(s)), u.T), W)
def do_fastica(X):
    n, m = X.shape; p = float(m); g = _logcosh
    #black magic
    X *= sqrt(X.shape[1])
    #create w
    W = ones((n,n), float32)
    for i in range(n):
        for j in range(i):
            W[i,j] = random.random()
    #compute W
    maxIter = 1000
    for ii in range(maxIter):
        # -----------------nandian
        gwtx, g_wtx = g(dot(W, X))
        W1 = do_decorrelation(dot(gwtx, X.T) / p - g_wtx[:, newaxis] * W)
        lim = max( abs(abs(diag(dot(W1, W.T))) - 1) )
        W = W1
        if lim < 0.0001:
            break
    return W
# 纵向显示图示
def show_data2(T,S):
    for j in range(4):
        plt.subplot(4,1,j+1)
        plt.plot(T, [S[j, i] for i in range(S.shape[1])])
    plt.show()
# 横向显示图示
def show_data(T, S,j):
    plt.plot(T, [S[0,i] for i in range(S.shape[1])])
    plt.plot(T, [S[1,i] for i in range(S.shape[1])])
    plt.plot(T, [S[2, i] for i in range(S.shape[1])])
    plt.plot(T, [S[3, i] for i in range(S.shape[1])])
    # plt.savefig("E:\BCI\picture\ICA\ICA"+str(j)+".png")
    plt.show()
def abs_value(x):
    a1 = abs(max(x) - min(x))
    return a1
def main():
    T, S, D = create_data()
    # data = sio.loadmat('D:\BCI_datal\data\ddatal3.mat')
    # dataL = data['datal3']
    # D = dataL[[2,4,5,12], :, 0, 0]
    # D = dataL[[8,9,10,12], :, 0, 0]
    # D2=ICA.ICA(dataL[[18,20,22,23], :, 0, 0])
    # D = dataL[[18,20,22,23], :, 0, 0]
    data = sio.loadmat('E:\BCI\physionet\mat\s1data\s1test\s1_03dataL.mat')
    dataL = data['data']
    D = dataL[0, [2,21,31,40], :]
    n_samples = 960
    T = linspace(0, 6, n_samples)
    Dwhiten, K = whiten(D)
    W = do_fastica(Dwhiten)
    #Sr: reconstructed source
    Sr = dot(dot(W, K), D)
    dd=[]
    for i in range(4):
        aa = abs_value(Sr[i])
        dd.append(aa)
    j=dd.index(max(dd))
    Sr[j]=0
    D1=dot(linalg.inv(dot(W, K)),Sr)
    show_data(T, D,1)
    # show_data(T, S)
    show_data(T, Sr,4)
    show_data(T, D1,3)
    # show_data(T, D2)
def eyesclose():
    data = sio.loadmat('C:\Users\lenovo\Desktop\eyesclose.mat')
    data = data['data']
    xn = data[0:5, 0:3200]
    fs=160
    n_samples = 3200
    T = linspace(0, 20, n_samples)
    Dwhiten, K = whiten(xn)
    W = do_fastica(Dwhiten)
    Sr = dot(dot(W, K), xn)
    # show_data(T,Sr,1)
    show_data2(T, Sr)
    # for i in range(6):
    #     plt.subplot(1, 6, i + 1)
    #     plt.psd(xn[i + 6, :], NFFT=256, pad_to=None, Fs=fs)
    # plt.show()
if __name__ == "__main__":
    # eyesclose()
    main()