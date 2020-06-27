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
    lista = sorted(os.listdir("."))

    acc_lista = []
    loss_lista = []

    for data in lista:
        if "epoch" in data:
            print("entro")
            acc_l = []
            loss_l = []
            file = open(data)
            for line in file:
                linea = line.rstrip('\n').split('\t')
                print(linea)
                acc_l.append(float(linea[0]))
                loss_l.append(float(linea[1]))
            file.close()
            acc_lista.append(acc_l)
            loss_lista.append(loss_l)
    return acc_lista, loss_lista


#lista_acc, lista_loss, lista_epoch_acc, lista_epoch_val_acc, lista_epoch_loss, lista_epoch_val_loss = ReadData()


def Multiplot(l1, tittle):
    largo = len(l1)
    if 1 <= largo <= 3:
        f = largo
        c = 1
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
        plt.plot(batchs, i, 'r-', label=tittle)
        base_y = min(i)
        sup_y = max(i)
        stepx = 20
        stepy = (sup_y-base_y)/10
        plt.xticks(np.arange(0, len(i)+stepx, stepx))
        plt.yticks(np.arange(base_y, sup_y+stepy, stepy))
        plt.grid()
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel(tittle)
        plt.title(f'{tittle} Epoch  {idx}')


if __name__ == "__main__":
    acc_lista, loss_lista = ReadData()
    Multiplot(acc_lista, tittle="Accuracy")
    Multiplot(loss_lista, tittle="Loss")
    plt.show()
