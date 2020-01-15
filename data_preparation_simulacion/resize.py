# -*- coding: utf-8 *-*
import extract
import leertxt
import cv2
import os
import shutil as st
#import divide as dv
#from . import rutas_data_preparation as rt
import rutas_data_preparation as rt
import sys
import random
from subprocess import call
from datetime import datetime

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


path_anormal = "Anomalous/"
path_normal = "Normal/"
path_a = "anomalous"
path_n = "normal"
cwd = ".."

paths = rt.Directorios(cwd=cwd)


def Crear_Directorios_Training():
    try:
        os.mkdir(paths.data_training_validation)
    except:
        st.rmtree(paths.data_training_validation)
        os.mkdir(paths.data_training_validation)

    try:
        os.mkdir(paths.data_training)
    except:
        st.rmtree(paths.data_training)
        os.mkdir(paths.data_training)

    try:
        os.mkdir(paths.data_temporal_normal)
    except:
        st.rmtree(paths.data_temporal_normal)
        os.mkdir(paths.data_temporal_normal)

    try:
        os.mkdir(paths.data_temporal_anomalous)
    except:
        st.rmtree(paths.data_temporal_anomalous)
        os.mkdir(paths.data_temporal_anomalous)

    try:
        os.mkdir(paths.data_validation)
    except:
        st.rmtree(paths.data_validation)
        os.mkdir(paths.data_validation)

    try:
        os.mkdir(paths.data_training_anomalous)
    except:
        st.rmtree(paths.data_training_anomalous)
        os.mkdir(paths.data_training_anomalous)

    try:
        os.mkdir(paths.data_validation_anomalous)
    except:
        st.rmtree(paths.data_validation_anomalous)
        os.mkdir(paths.data_validation_anomalous)

    try:
        os.mkdir(paths.data_training_normal)
    except:
        st.rmtree(paths.data_training_normal)
        os.mkdir(paths.data_training_normal)

    try:
        os.mkdir(paths.data_validation_normal)
    except:
        st.rmtree(paths.data_validation_normal)
        os.mkdir(paths.data_validation_normal)
# --------------


def main():
    Crear_Directorios_Training()
    p1 = os.path.join(paths.data_training_validation+"_exp",'training','normal')
    p2 = os.path.join(paths.data_training_validation+"_exp",'training','anomalous')
    p3 = os.path.join(paths.data_training_validation+"_exp",'validation','normal')
    p4 = os.path.join(paths.data_training_validation+"_exp",'validation','anomalous')


    l1 = os.listdir(p1)
    l2 = os.listdir(p2)
    l3 = os.listdir(p3)
    l4 = os.listdir(p4)
    for i in l1:
        print(p1,i,f"to {os.path.join(paths.data_training_normal,i)}")
        img = cv2.imread(os.path.join(p1,i), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
        cv2.imwrite(os.path.join(paths.data_training_normal,i),img)
    for i in l2:
        print(p2,i,f"to {os.path.join(paths.data_training_anomalous,i)}")
        img = cv2.imread(os.path.join(p2,i), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
        cv2.imwrite(os.path.join(paths.data_training_anomalous,i),img)
    for i in l3:
        print(p3,i,f"to {os.path.join(paths.data_validation_normal,i)}")
        img = cv2.imread(os.path.join(p3,i), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
        cv2.imwrite(os.path.join(paths.data_validation_normal,i),img)
    for i in l4:
        print(p4,i,f"to {os.path.join(paths.data_validation_anomalous,i)}")
        img = cv2.imread(os.path.join(p4,i), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
        cv2.imwrite(os.path.join(paths.data_validation_anomalous,i),img)



if __name__ == '__main__':
    main()
