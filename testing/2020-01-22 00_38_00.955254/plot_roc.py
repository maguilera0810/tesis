from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

def ReadData():
    y_test, y_pred_r, y_pred_i, y_pred_f = [], [], [], []
    archivo = open("roc.txt", "r")
    for line in archivo:
        linea = line.rstrip('\n').split('\t')
        y_test.append(float(linea[0]))
        y_pred_r.append(float(linea[1]))
        y_pred_i.append(float(linea[2]))
        y_pred_f.append(float(linea[3]))
    return y_test, y_pred_r, y_pred_i, y_pred_f


def plot_roc():
    y_test, y_pred_r, y_pred_i, y_pred_f = ReadData()
    fpr_r, tpr_r, thresholds_r = roc_curve(y_test, y_pred_r)
    auc_r = auc(fpr_r, tpr_r)
    print(auc_r)
    fpr_i, tpr_i, thresholds_i = roc_curve(y_test, y_pred_i)
    auc_i = auc(fpr_i, tpr_i)
    print(auc_i)
    fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred_f)
    auc_f = auc(fpr_f, tpr_f)
    print(auc_f)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_r, tpr_r, 'r', label='Resnet50 (area = {:.3f})'.format(auc_r))
    plt.plot(fpr_i, tpr_i, 'g',
             label='InceptionV3 (area = {:.3f})'.format(auc_i))
    plt.plot(fpr_f, tpr_f, 'b',
             label='FrankensNet (area = {:.3f})'.format(auc_f))
    plt.xlabel('False positive rate',fontsize=30)
    plt.ylabel('True positive rate',fontsize=30)
    plt.title('ROC curve',fontsize=30)
    plt.xticks(np.arange(0, 1.05, 0.05))
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid()
    #plt.legend()
    plt.legend(loc='best',fontsize=20)
    plt.show()
if __name__ == "__main__":
    plot_roc()
