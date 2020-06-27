from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt
import os


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

    names = [
        "resnet50.txt",
        "inceptionv3.txt",
        "frankensnet.txt"
    ]
    lista_fp_r = []
    lista_tp_r = []
    lista_fn_r = []
    lista_tn_r = []

    lista_fp_i = []
    lista_tp_i = []
    lista_fn_i = []
    lista_tn_i = []

    lista_fp_f = []
    lista_tp_f = []
    lista_fn_f = []
    lista_tn_f = []
    try:
        file = open(names[0], "r")
        for line in file:
            line = line.rstrip('\n').split('\t')
            # lista_fp_r.append(float(line[0])/(float(line[0])+float(line[3])))
            # lista_tp_r.append(float(line[1])/(float(line[1])+float(line[2])))
            # lista_fp_r.append(float(line[0])+float(line[2]))
            # lista_tp_r.append(float(line[1])+float(line[3]))
            lista_fp_r.append(float(line[0]))
            lista_tp_r.append(float(line[1]))
            lista_fn_r.append(float(line[2]))
            lista_tn_r.append(float(line[3]))
        file.close()
        # print(zip(lista_fp_r,lista_tp_r))
        max_fp_r = max(lista_fp_r)
        max_tp_r = max(lista_tp_r)
        max_fn_r = max(lista_fn_r)
        max_tn_r = max(lista_tn_r)
        resnet = max_fp_r, max_tp_r, max_fn_r, max_tn_r

        for i in range(len(lista_fp_r)):
            lista_fp_r[i] /= max_fp_r
            lista_tp_r[i] /= max_tp_r
        # print(zip(lista_fp_r,lista_tp_r))
    except:
        print("no data resnet50")
        resnet = None

    try:
        file = open(names[1], "r")
        for line in file:
            line = line.rstrip('\n').split('\t')
            # lista_fp_i.append(float(line[0])/(float(line[0])+float(line[3])))
            # lista_tp_i.append(float(line[1])/(float(line[1])+float(line[2])))
            # lista_fp_i.append(float(line[0])+float(line[2]))
            # lista_tp_i.append(float(line[1])+float(line[3]))
            lista_fp_i.append(float(line[0]))
            lista_tp_i.append(float(line[1]))
            lista_fn_i.append(float(line[2]))
            lista_tn_i.append(float(line[3]))
        file.close()
        max_fp_i = max(lista_fp_i)
        max_tp_i = max(lista_tp_i)
        max_fn_i = max(lista_fn_i)
        max_tn_i = max(lista_tn_i)
        inception = max_fp_i, max_tp_i, max_fn_i, max_tn_i

        for i in range(len(lista_fp_i)):
            lista_fp_i[i] /= max_fp_i
            lista_tp_i[i] /= max_tp_i
    except:
        print("no data inceptionv3")
        inception = None
    try:
        file = open(names[2], "r")
        for line in file:
            line = line.rstrip('\n').split('\t')
            # lista_fp_f.append(float(line[0])/(float(line[0])+float(line[3])))
            # lista_tp_f.append(float(line[1])/(float(line[1])+float(line[2])))
            # lista_fp_f.append(float(line[0])+float(line[2]))
            # lista_tp_f.append(float(line[1])+float(line[3]))
            lista_fp_f.append(float(line[0]))
            lista_tp_f.append(float(line[1]))
            lista_fn_f.append(float(line[2]))
            lista_tn_f.append(float(line[3]))
        file.close()
        max_fp_f = max(lista_fp_f)
        max_tp_f = max(lista_tp_f)
        max_fn_f = max(lista_fn_f)
        max_tn_f = max(lista_tn_f)
        frankensnet = max_fp_f, max_tp_f, max_fn_f, max_tn_f
        for i in range(len(lista_fp_f)):
            lista_fp_f[i] /= max_fp_f
            lista_tp_f[i] /= max_tp_f
    except:
        print("no data frankensnet")
        frankensnet = None

    print(len(lista_fp_r), len(lista_tp_r))
    print(len(lista_fp_i), len(lista_tp_i))
    print(len(lista_fp_f), len(lista_tp_f))
    print(resnet)
    print(inception)
    print(frankensnet)
    return lista_fp_r, lista_tp_r, lista_fp_i, lista_tp_i, lista_fp_f, lista_tp_f, resnet, inception, frankensnet


def Plot():
    lista_fp_r, lista_tp_r, lista_fp_i, lista_tp_i, lista_fp_f, lista_tp_f, resnet, inception, frankensnet = ReadData()
    print(resnet)
    print(inception)
    print(frankensnet)
    puntos = [i for i in range(len(lista_fp_r))]
    plt.figure()
    plt.plot([0, 1], [0, 1], 'b--')
    if lista_fp_r != []:
        # print(zip(lista_fp_r,lista_tp_r))
        plt.plot(lista_fp_r, lista_tp_r, 'r')
    if lista_fp_i != []:
        # print(list(zip(lista_fp_i,lista_tp_i)))
        plt.plot(lista_fp_i, lista_tp_i, 'b')
    if lista_fp_f != []:
        # print(zip(lista_fp_f,lista_tp_f))
        plt.plot(lista_fp_f, lista_tp_f, 'g')
    plt.title('ROC curve')
    step = 0.05
    plt.xticks(np.arange(0, 1+step, step))
    plt.yticks(np.arange(0, 1+step, step))
    plt.grid()
    fpr = []
    vpr = []
    if resnet != None:
        fp_r, tp_r, fn_r, tn_r = resnet
        fpr_r = fp_r/(fp_r+tn_r)
        vpr_r = tp_r/(tp_r+fn_r)
        fpr.append(fpr_r)
        vpr.append(vpr_r)
    if inception != None:
        fp_i, tp_i, fn_i, tn_i = inception
        fpr_i = fp_i/(fp_i+tn_i)
        vpr_i = tp_i/(tp_i+fn_i)
        fpr.append(fpr_i)
        vpr.append(vpr_i)
    if frankensnet != None:
        fp_f, tp_f, fn_f, tn_f = frankensnet
        fpr_f = fp_f/(fp_f+tn_f)
        vpr_f = tp_f/(tp_f+fn_f)
        fpr.append(fpr_f)
        vpr.append(vpr_f)
    labels = ["Resnet50", "Inception", "FrankensNet"]
    plt.figure()
    plt.plot([0, 1], [0, 1], 'b--')
    plt.scatter(fpr, vpr)
    step = 0.05
    plt.xticks(np.arange(0, 1+step, step))
    plt.yticks(np.arange(0, 1+step, step))
    for i, txt in enumerate(labels):
        plt.annotate(txt, (fpr[i], vpr[i]))
    plt.title('ROC space')
    plt.grid()

    return resnet, inception, frankensnet


def RocSpace(resnet, inception, frankensnet):
    fp_r, tp_r, fn_r, tn_r = resnet
    fp_i, tp_i, fn_i, tn_i = inception
    fp_f, tp_f, fn_f, tn_f = frankensnet
    fpr_r = fp_r/(fp_r+tn_r)
    vpr_r = tp_r/(tp_r+fn_r)
    fpr_i = fp_i/(fp_i+tn_i)
    vpr_i = tp_i/(tp_i+fn_i)
    fpr_f = fp_f/(fp_f+tn_f)
    vpr_f = tp_f/(tp_f+fn_f)
    fpr = [fpr_r, fpr_i, fpr_f]
    vpr = [vpr_r, vpr_i, vpr_f]
    plt.figure()
    plt.scatter(fpr, vpr)
    plt.title('ROC space')
    plt.grid()
    plt.show()
    pass


if __name__ == "__main__":
    resnet, inception, frankensnet = Plot()
    # RocSpace()
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
