from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
import cv2
import leertxt
import extract_texting
import shutil as st
import rutas_data_preparation as rt
from datetime import datetime
from time import time
from tkinter import *
from plots import Plot
#import h5py
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 224, 224

cwd = ".."
PATHS = rt.Directorios(cwd=cwd)


modelos = [
    os.path.join(PATHS.checkpoints_resnet, "ResNet50_model_weights.h5"),
    os.path.join(PATHS.checkpoints_inception, "InceptionV3_model_weights.h5"),
    os.path.join(PATHS.checkpoints_frankensnet, "FrankensNet_model_weights.h5")
]


prueba1 = 1
prueba2 = 1
prueba3 = 1


y_test = []

y_pred_r = []
y_pred_i = []
y_pred_f = []


if prueba1:
    classifier_resnet = load_model(modelos[0], compile=False)
    print("\n\n##############resnet#################\n\n")
if prueba2:
    #f = h5py.File(modelos[1], "r")
    classifier_inception = load_model(modelos[1], compile=False)
    print("\n\n###############inception################\n\n")
if prueba3:
    classifier_frankensnet = load_model(modelos[2], compile=False)
    print("\n\n################frankensnet###############\n\n")

# classifier_frankensnet= load_model(modelos[2])


def GuardarDatos(lr, li, lf, yt, ypr, ypi, ypf):
    file_r = open("roc.txt", "w")
    for i in range(len(yt)):
        file_r.write(f"{yt[i]}\t{ypr[i]}\t{ypi[i]}\t{ypf[i]}\n")
    file_r.close()

    if lr != []:
        file_r = open("resnet50.txt", "w")
        for r in lr:

            file_r.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")
        file_r.close()
    if li != []:
        file_i = open("inceptionv3.txt", "w")
        for i in li:

            file_i.write(f"{i[0]}\t{i[1]}\t{i[2]}\t{i[3]}\n")
        file_i.close()
    if lf != []:
        file_f = open("frankensnet.txt", "w")
        for f in lf:

            file_f.write(f"{f[0]}\t{f[1]}\t{f[2]}\t{f[3]}\n")
        file_f.close()
    return [], [], [], [], [], [], []


def parameters(tp, fp, tn, fn):
    lista = []
    try:
        lista.append(tp_i/(tp_i+fp_i))
    except:
        lista.append(0)
    try:
        lista.append(fp_i/(fp_i+tn_i))
    except:
        lista.append(0)
    try:
        lista.append(tn_i/(tn_i+fn_i))
    except:
        lista.append(0)
    try:
        lista.append(fn_i/(fn_i+tp_i))
    except:
        lista.append(0)
    return lista


fps = 30
f_n, f_a = leertxt.leer(
    PATHS.normal_training_data_txt,
    PATHS.anomalous_training_data_txt
)


def main():
    t_r, t_i, t_f, conta = 0, 0, 0, 0
    if prueba1:
        parameters_r = fp_r, tp_r, fn_r, tn_r = 0, 0, 0, 0
    else:
        parameters_r = None
    if prueba2:
        parameters_i = fp_i, tp_i, fn_i, tn_i = 0, 0, 0, 0
    else:
        parameters_i = None
    if prueba3:
        parameters_f = fp_f, tp_f, fn_f, tn_f = 0, 0, 0, 0
    else:
        parameters_f = None
    path = "screens"
    lr, li, lf, y_test, y_pred_r, y_pred_i, y_pred_f = [], [], [], [], [], [], []

    for idx, video in enumerate(f_a):
        print(vars(video))
        try:
            os.mkdir(path)
        except:
            st.rmtree(path)
            os.mkdir(path)
        try:
            video_capture = cv2.VideoCapture(
                os.path.join(PATHS.anomalous_data_set, video.name))
            # print(os.path.join(PATHS.anomalous_data_set, video.name))
        except:
            print(
                "ERROR CV2 ANOMALOUS-----------------------------------")
            continue
        i = 0

        while True:  # fps._numFrames < 120

            check, frame = video_capture.read()  # get current frame

            if check:
                f_name = os.path.join(path, f"{i}_alpha.png")
                cv2.imwrite(filename=f_name, img=frame)

                if not extract_texting.checkear(video.tramos_no_usar, fps, i):

                    if i % 5 == 0:
                        # test_image = image.load_img(f_name, target_size=(255, 255))
                        conta += 1
                        test_image = image.load_img(
                            f_name, target_size=(WIDTH, HEIGHT))
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(
                            test_image, axis=0)

                        if prueba1:
                            to = time()
                            result_resnet = classifier_resnet.predict(
                                test_image)
                            t_r += (time()-to)
                            data_resnet = {
                                "anomalous": result_resnet[0][0], "normal": result_resnet[0][1]}
                            y_pred_r.append(data_resnet.get("anomalous"))
                        else:
                            data_resnet = None
                        if prueba2:
                            to = time()
                            result_inception = classifier_inception.predict(
                                test_image)
                            t_i += (time()-to)
                            data_inception = {
                                "anomalous": result_inception[0][0], "normal": result_inception[0][1]}
                            y_pred_i.append(data_inception.get("anomalous"))
                        else:
                            data_inception = None
                        if prueba3:
                            to = time()
                            result_frankensnet = classifier_frankensnet.predict(
                                test_image)
                            t_f += (time()-to)
                            data_frankensnet = {
                                "anomalous": result_frankensnet[0][0], "normal": result_frankensnet[0][1]}
                            y_pred_f.append(data_frankensnet.get("anomalous"))
                        else:
                            data_frankensnet = None

                        if extract_texting.checkear(video.tramosAnomalos, fps, i):
                            if prueba1:
                                if data_resnet.get('anomalous') > data_resnet.get('normal'):
                                    tp_r += 1
                                else:
                                    fn_r += 1

                            if prueba2:
                                if data_inception.get('anomalous') > data_inception.get('normal'):
                                    tp_i += 1
                                else:
                                    fn_i += 1
                            if prueba3:
                                if data_frankensnet.get('anomalous') > data_frankensnet.get('normal'):
                                    tp_f += 1
                                else:
                                    fn_f += 1
                            y_test.append(1)
                        else:
                            if prueba1:
                                if data_resnet.get('anomalous') > data_resnet.get('normal'):
                                    fp_r += 1
                                else:
                                    tn_r += 1
                            if prueba2:
                                if data_inception.get('anomalous') > data_inception.get('normal'):
                                    fp_i += 1
                                else:
                                    tn_i += 1
                            if prueba3:
                                if data_frankensnet.get('anomalous') > data_frankensnet.get('normal'):
                                    fp_f += 1
                                else:
                                    tn_f += 1
                            y_test.append(0)
                        if prueba1:
                            parameters_r = fp_r, tp_r, fn_r, tn_r
                            lr.append(parameters_r)
                            res_r = f"fp: {fp_r} fn: {fn_r} tp: {tp_r} tn: {tn_r} total: {fp_r + fn_r + tp_r + tn_r} | "
                        else:
                            res_r = ""
                        if prueba2:
                            parameters_i = fp_i, tp_i, fn_i, tn_i
                            li.append(parameters_i)
                            res_i = f"fp: {fp_i} fn: {fn_i} tp: {tp_i} tn: {tn_i} total: {fp_i + fn_i + tp_i + tn_i} | "
                        else:
                            res_i = ""
                        if prueba3:
                            parameters_f = fp_f, tp_f, fn_f, tn_f
                            lf.append(parameters_f)
                            res_f = f"fp: {fp_f} fn: {fn_f} tp: {tp_f} tn: {tn_f} total: {fp_f + fn_f + tp_f + tn_f} | "
                        else:
                            res_f = ""
                    res = res_r + res_i + res_f
                    print(res, end="\r")
                i += 1
            else:
                break
        video_capture.release()
        GuardarDatos(lr, li, lf, y_test, y_pred_r, y_pred_i, y_pred_f)

    if False:
        for video in f_n:
            break  # --------------------borrar
            print(vars(video))
            try:
                os.mkdir(path)
            except:
                st.rmtree(path)
                os.mkdir(path)
            try:
                video_capture = cv2.VideoCapture(
                    os.path.join(PATHS.normal_data_set, video.name))
            except:
                print(
                    "ERROR CV2 ANOMALOUS-----------------------------------")
                continue
            i = 0
            while True:  # fps._numFrames < 120

                check, frame = video_capture.read()  # get current frame

                if check:

                    f_name = os.path.join(path, f"{i}_alpha.png")
                    # write frame image to file
                    cv2.imwrite(filename=f_name, img=frame)
                    if i % 25 == 0:
                        test_image = image.load_img(
                            f_name, target_size=(WIDTH, WIDTH))
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(
                            test_image, axis=0)

                        if prueba1:
                            result_resnet = classifier_resnet.predict(
                                test_image)
                            data_resnet = {
                                "anomalous": result_resnet[0][0], "normal": result_resnet[0][1]}
                            if data_resnet.get('anomalous') > data_resnet.get('normal'):
                                fp_r += 1
                            else:
                                tn_r += 1
                            parameters_r = tp_r, fp_r, tn_r, fn_r
                            lr.append(parameters_r)
                            res_r = f"fp: {fp_r} fn: {fn_r} tp: {tp_r} tn: {tn_r} total: {fp_r + fn_r + tp_r + tn_r} | "
                        else:
                            data_resnet = None
                            res_r = ""

                        if prueba2:
                            result_inception = classifier_inception.predict(
                                test_image)
                            data_inception = {
                                "anomalous": result_inception[0][0], "normal": result_inception[0][1]}
                            if data_inception.get('anomalous') > data_inception.get('normal'):
                                fp_i += 1
                            else:
                                tn_i += 1
                            parameters_i = tp_i, fp_i, tn_i, fn_i
                            li.append(parameters_i)
                            res_i = f"fp: {fp_i} fn: {fn_i} tp: {tp_i} tn: {tn_i} total: {fp_i + fn_i + tp_i + tn_i} | "
                        else:
                            data_inception = None
                            res_i = ""

                        if prueba3:
                            result_frankensnet = classifier_frankensnet.predict(
                                test_image)
                            data_frankensnet = {
                                "anomalous": result_frankensnet[0][0], "normal": result_frankensnet[0][1]}
                            if data_frankensnet.get('anomalous') > data_frankensnet.get('normal'):
                                fp_f += 1
                            else:
                                tn_f += 1
                            parameters_f = tp_f, fp_f, tn_f, fn_f
                            lf.append(parameters_f)
                            res_f = f"fp: {fp_f} fn: {fn_f} tp: {tp_f} tn: {tn_f} total: {fp_f + fn_f + tp_f + tn_f} | "
                        else:
                            data_frankensnet = None
                            res_f = ""
                        res = res_r + res_i + res_f
                        print(res, end="\r")
                    i += 1
                else:
                    break
            video_capture.release()
            GuardarDatos(lr, li, lf)

    try:
        os.mkdir(path)
    except:
        st.rmtree(path)
        os.mkdir(path)

    if prueba1:
        fpr_r, tpr_r, thresholds_r = roc_curve(y_test, y_pred_r)
        auc_r = auc(fpr_r, tpr_r)
        t_r /= conta
        print(f"Time: {t_r} pruebas: {conta}")
        print(auc_r)
    if prueba2:
        fpr_i, tpr_i, thresholds_i = roc_curve(y_test, y_pred_i)
        auc_i = auc(fpr_i, tpr_i)
        t_i /= conta
        print(f"Time: {t_i}")
        print(auc_i)
    if prueba3:
        fpr_f, tpr_f, thresholds_f = roc_curve(y_test, y_pred_f)
        auc_f = auc(fpr_f, tpr_f)
        t_f /= conta
        print(f"Time: {t_f}")
        print(auc_f)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    if prueba1:
        plt.plot(fpr_r, tpr_r, 'g',
                 label='Resnet50 (area = {:.3f})'.format(auc_r))
    if prueba2:
        plt.plot(fpr_i, tpr_i, 'r',
                 label='InceptionV3 (area = {:.3f})'.format(auc_i))
    if prueba3:
        plt.plot(fpr_f, tpr_f, 'b',
                 label='FrankensNet (area = {:.3f})'.format(auc_f))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


main()
Plot()
# ------------------------------------------------------------------------------------------------------
