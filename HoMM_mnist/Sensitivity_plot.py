from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.size']=10
plt.rcParams['axes.labelsize']=12
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['axes.titlesize']=12
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['legend.fontsize']=10
plt.rcParams['figure.titlesize']=12


def ParameterSentivity():
    lambda1=[1,10,50,100,500,1000,10000,100000]
    acc_mnist_lambda1=[0.70,0.86,0.962,0.965,0.970,0.972,0.933,0.916]
    acc_amazon_lambda1=[0.825,0.832,0.846,0.849,0.853,0.851,0.837,0.762]

    plt.figure(figsize=(4,3))
    plt.semilogx(lambda1,acc_mnist_lambda1,marker="s",c="b",lw=3,ms=8,label="SVHN-MNIST")
    plt.hlines(0.895,1,100000,color='b',linestyles='dashed')
    plt.semilogx(lambda1, acc_amazon_lambda1, marker="o", c="r", lw=3, ms=8, label="A-D")
    plt.hlines(0.787, 1, 100000, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{ssc}$")
    plt.ylabel("Accuracy")
    # plt.xticks([1,10,50,100,500,1000,10000,100000],[1,2,3,4,5,6,7,8])
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()

    lambda2=[0.00001,0.0001,0.001,0.003,0.01,0.03,0.1,0.3]
    acc_mnist_lambda2=[0.962,0.970,0.972,0.966,0.952,0.923,0.861,0.648]
    acc_amazon_lambda2=[0.812,0.834,0.845,0.852,0.854,0.848,0.842,0.788]
    plt.figure(figsize=(4, 3))
    plt.semilogx(lambda2, acc_mnist_lambda2, marker="s", c="b", lw=3,ms=8,label="SVHN-MNIST")
    plt.semilogx(lambda2, acc_amazon_lambda2, marker="o", c="r", lw=3, ms=8, label="A-D")
    plt.hlines(0.942, 0, 0.3, color='b', linestyles='dashed')
    plt.hlines(0.763, 0, 0.3, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{intra}$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()



    lambda3=[0.000001,0.00001,0.0001,0.0003,0.0005,0.001,0.003,0.01]
    acc_mnist_lambda3=[0.945,0.946,0.947,0.959,0.964,0.973,0.955,0.82]
    acc_amazon_lambda3=[0.804,0.818,0.844,0.853,0.864,0.829,0.812,0.772]
    plt.figure(figsize=(4, 3))
    plt.semilogx(lambda3, acc_mnist_lambda3, marker="s", c="b", lw=3,ms=8,label="SVHN-MNIST")
    plt.hlines(0.942, 0, 0.01, color='b', linestyles='dashed')
    plt.semilogx(lambda3, acc_amazon_lambda3, marker="o", c="r", lw=3,ms=8,label="A-D")
    plt.hlines(0.763, 0, 0.01, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{inter}$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()



def ParameterSentivity_HoMM():
    lambda_d=[1,10,100,1000,10000,100000,1000000,10000000]
    acc_mnist_lambda_d=[0.69,0.86,0.93,0.957,0.972,0.958,0.933,0.916]
    acc_amazon_lambda_d=[0.776,0.832,0.88,0.90,0.861,0.821,0.73,0.66]
    plt.figure(figsize=(4,3))
    plt.semilogx(lambda_d,acc_mnist_lambda_d,marker="s",c="b",lw=3,ms=8,label="SVHN-MNIST")
    plt.semilogx(lambda_d, acc_amazon_lambda_d, marker="o", c="r", lw=3, ms=8, label="A-W")
    # plt.hlines(0.895, 1, 10000000, color='b', linestyles='dashed')
    # plt.hlines(0.793, 1, 10000000, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{d}$")
    plt.ylabel("Accuracy")
    # plt.xticks([1,10,50,100,500,1000,10000,100000],[1,2,3,4,5,6,7,8])
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()



    lambda_dc=[0.00001,0.0001,0.001,0.01,0.03,0.1,0.3,1.0]
    acc_mnist_lambda_dc=[0.972,0.971,0.974,0.975,0.978,0.988,0.981,0.912]
    acc_amazon_lambda_dc=[0.904,0.904,0.905,0.904,0.909,0.917,0.902,0.858]
    plt.figure(figsize=(4, 3))
    plt.semilogx(lambda_dc, acc_mnist_lambda_dc, marker="s", c="b", lw=3,ms=8,label="SVHN-MNIST")
    plt.semilogx(lambda_dc, acc_amazon_lambda_dc, marker="o", c="r", lw=3, ms=8, label="A-W")
    plt.hlines(0.972, 0.00001, 1.0, color='b', linestyles='dashed')
    plt.hlines(0.905, 0.00001, 1.0, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{dc}$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()


    N = [10,100,1000, 3000, 10000, 30000, 100000, 300000]
    acc_mnist_lambda_N = [0.872,0.943, 0.959, 0.962, 0.968, 0.971, 0.970, 0.972]
    acc_amazon_lambda_N = [0.74, 0.842, 0.878, 0.883, 0.889, 0.895, 0.895, 0.896]
    plt.figure(figsize=(4, 3))
    plt.semilogx(N, acc_mnist_lambda_N, marker="s", c="b", lw=3, ms=8, label="SVHN-MNIST")
    plt.semilogx(N, acc_amazon_lambda_N, marker="o", c="r", lw=3, ms=8, label="A-W")
    # plt.hlines(0.895, 100, 500000, color='b', linestyles='dashed')
    # plt.hlines(0.793, 100, 500000, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{dc}$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()

    delta = [0.5,0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    acc_mnist_lambda_delta = [0.931,0.952, 0.966, 0.970, 0.982, 0.987, 0.986, 0.988, 0.983]
    acc_amazon_lambda_delta = [0.88,0.912, 0.914, 0.917, 0.914, 0.915, 0.915, 0.908, 0.907]
    plt.figure(figsize=(4, 3))
    plt.plot(delta, acc_mnist_lambda_delta, marker="s", c="b", lw=3, ms=8, label="SVHN-MNIST")
    plt.plot(delta, acc_amazon_lambda_delta, marker="o", c="r", lw=3, ms=8, label="A-W")
    plt.hlines(0.972, 0.5, 0.95, color='b', linestyles='dashed')
    plt.hlines(0.905, 0.5, 0.95, color='r', linestyles='dashed')
    plt.xlabel(r"$\lambda_{dc}$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(loc="lower center")
    plt.show()




if __name__=="__main__":
    # ParameterSentivity()
    ParameterSentivity_HoMM()
