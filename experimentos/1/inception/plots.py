from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import rutas_data_preparation as rt
import matplotlib.pyplot as plt
import os
#import seaborn as sns

cwd = ""
paths = rt.Directorios(os.path.join(cwd))
print(paths.exp_final_frankensnet_batchs)


def suavizar(x, y):
    if len(x) > 2:
        T = np.array(x)
        # 300 represents number of points to make between T.min and T.max
        x_suave = np.linspace(T.min(), T.max(), 300)
        spl = make_interp_spline(T, y, k=3)  # BSpline object
        y_suave = spl(x_suave)
        return x_suave, y_suave
    else:
        return x, y


def ReadData():
    # print(paths.batch_data)
    resnet = sorted(os.listdir("inception",winner))
    print(resnet)
    inception = sorted(os.listdir(paths.exp_final_inception_batchs))
    print(inception)
    frankensnet = sorted(os.listdir(paths.exp_final_frankensnet_batchs))
    print(frankensnet)

    modelos = zip(resnet, inception, frankensnet)
    acc_lista_r = []
    loss_lista_r = []
    acc_lista_i = []
    loss_lista_i = []
    acc_lista_f = []
    loss_lista_f = []

    for res, inc, frank in modelos:
        acc_l = []
        loss_l = []
        file = open(os.path.join(paths.exp_final_resnet_batchs, res))
        for line in file:
            linea = line.rstrip('\n').split('\t')
            acc_l.append(float(linea[0]))
            loss_l.append(float(linea[1]))
        file.close()
        acc_lista_r.append(acc_l)
        loss_lista_r.append(loss_l)
        acc_l = []
        loss_l = []
        file = open(os.path.join(paths.exp_final_inception_batchs, inc))
        for line in file:
            linea = line.rstrip('\n').split('\t')
            acc_l.append(float(linea[0]))
            loss_l.append(float(linea[1]))
        file.close()
        acc_lista_i.append(acc_l)
        loss_lista_i.append(loss_l)
        acc_l = []
        loss_l = []
        file = open(os.path.join(paths.exp_final_frankensnet_batchs, frank))
        for line in file:
            linea = line.rstrip('\n').split('\t')
            acc_l.append(float(linea[0]))
            loss_l.append(float(linea[1]))
        file.close()
        acc_lista_f.append(acc_l)
        loss_lista_f.append(loss_l)
        # print(batchs1)
        print(acc_lista_r[0] == acc_lista_i[0])
    return acc_lista_r, loss_lista_r, acc_lista_i, loss_lista_i, acc_lista_f, loss_lista_f


#lista_acc, lista_loss, lista_epoch_acc, lista_epoch_val_acc, lista_epoch_loss, lista_epoch_val_loss = ReadData()


def Multiplot1(l1, tittle):
    largo = len(l1)
    if 1 <= largo <= 3:
        f = 1
        c = largo
    elif 4 <= largo <= 6:
        f = 2
        if largo == 4:
            c = 2
        else:
            c = 3
    elif largo in [7, 9]:
        f = 3
        c = 3
    elif largo == 8:
        f = 2
        c = 4
    else:
        f = 3
        c = 4
    plt.figure()
    for idx, i in enumerate(l1):
        batchs = [j for j in range(len(i))]
        #batchs, i = suavizar(batchs, i)
        #print(f,c,idx +1)
        plt.subplot(f, c, idx+1)
        plt.plot(batchs, i, 'r', label='ejemplo')
        plt.xlabel('Batchs')
        plt.ylabel(tittle)
        plt.title(f'{tittle} {idx}')


def Multiplot2(l_r, l_i, l_f, tittle):
    largo = len(l_r)
    f = 5
    c = 1
    modelos = zip(l_r, l_i, l_f)
    print(l_r[0], l_i[0], l_f[0], "+++")
    print(len(l_r[0]), len(l_i), len(l_f))
    idx = 0
    for lr, li, lf in modelos:
        print(idx)
        batchs = [j for j in range(len(lr))]
        #batchs, i = suavizar(batchs, i)
        #print(f,c,idx +1)
        #plt.subplot(f, c, idx+1)
        plt.plot(batchs, lr, 'r-', label='Resnet50')
        plt.plot(batchs, li, 'g--', label='InceptionV3')
        plt.plot(batchs, lf, 'b:', label='FrankensNet')
        # plt.xlabel('Batchs')
        # plt.ylabel(tittle)
        #step = 0.05
        #plt.xticks(np.arange(0, 1+step, step))
        #plt.yticks(np.arange(0, 1+step, step))
        # plt.grid()
        plt.title(f'{tittle} {idx}')
        plt.figure()
        idx += 1


def Multiplot(l_r, l_i, l_f, tittle):
    largo = len(l_r)
    f = 5
    c = 1
    modelos = zip(l_r, l_i, l_f)
    idx = 0
    for lr, li, lf in modelos:
        # for i in range(100):
            # print(lr[i],li[i],lf[i])
        print(idx)
        batchs = [j for j in range(len(lr))]
        #batchs, i = suavizar(batchs, i)
        #print(f,c,idx +1)
        #plt.subplot(f, c, idx+1)
        batchs = [j/(len(li)-1) for j in range(len(li))]
        plt.plot(batchs, li, 'g:', label='InceptionV3')
        # plt.figure()
        batchs = [j/(len(lr)-1) for j in range(len(lr))]
        plt.plot(batchs, lr, 'r-', label='Resnet50')
        # plt.figure()
        batchs = [j/(len(lf)-1) for j in range(len(lf))]
        plt.plot(batchs, lf, 'b-', label='FrankensNet')
        # plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        base_y = min(min(lr), min(li), min(lf))
        sup_y = max(max(lr), max(li), max(lf))
        if sup_y >= 1:
            sup_y = 1
        stepx = 0.05
        stepy = (sup_y-base_y)/30
        plt.xticks(np.arange(0, 1+stepx, stepx))
        plt.yticks(np.arange(base_y, sup_y+stepy, stepy))
        plt.grid()
        plt.legend()
        plt.title(f'{tittle} Epoch{idx}')
        plt.figure()
        idx += 1


if __name__ == "__main__":
    acc_lista_r, loss_lista_r, acc_lista_i, loss_lista_i, acc_lista_f, loss_lista_f = ReadData()
    Multiplot(acc_lista_r, acc_lista_i, acc_lista_f, tittle="Accuracy")
    Multiplot(loss_lista_r, loss_lista_i, loss_lista_f, tittle="Loss")
    plt.show()
"""     T = np.array([6, 7, 8, 9, 10, 11, 12])
    power = np.array([1.53E+03, 5.92E+02, 2.04E+02,
                      7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])
    plt.plot(T, power)
    plt.figure()

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(T.min(), T.max(), 300)
    spl = make_interp_spline(T, power, k=3)  # BSpline object
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth)
    plt.show() """
