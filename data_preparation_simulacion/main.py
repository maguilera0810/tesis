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


def Cortar_videos(video, t0, t1):
    if bool(video.tipo):
        src = os.path.join(rt.anomalous_data_set, video.name)
    else:
        src = os.path.join(rt.normal_data_set, video.name)
        call(["ffmpeg", "-i", src, "-ss", t0, "-t", t1, "-c", "copy", src])


def DataRandom():
    files_a = sorted(os.listdir(paths.data_temporal_anomalous))
    files_n = sorted(os.listdir(paths.data_temporal_normal))

    print("Randomizando los elementos ...")
    for i in range(300):
        random.shuffle(files_a)
        random.shuffle(files_n)
    print("Hecho")
    lfa = len(files_a)
    lfn = len(files_n)
    per_t = 0.9
    per_v = 1 - per_t
    per_training_a = lfa * per_t
    per_training_n = lfn * per_t

    print("Anomalous")
    ct, cv = 0, 0
    for idx, file in enumerate(files_a):
        img = cv2.imread(os.path.join(
            paths.data_temporal_anomalous, file), cv2.IMREAD_UNCHANGED)
        if idx+1 <= per_training_a:
            cv2.imwrite(os.path.join(paths.data_training_anomalous, file), img)
            ct += 1
        else:
            cv2.imwrite(os.path.join(
                paths.data_validation_anomalous, file), img)
            cv += 1
        print(
            f"Anomalous frames: Training = {ct} Validation = {cv} T = {round(ct/lfa,4)} V = {round(cv/lfa,4)}", end="\r")
    print(
        f"Anomalous frames: Training = {ct} Validation = {cv} T = {round(ct/lfa,4)} V = {round(cv/lfa,4)}")
    ct, cv = 0, 0
    print("Normal")
    for idx, file in enumerate(files_n):
        img = cv2.imread(os.path.join(
            paths.data_temporal_normal, file), cv2.IMREAD_UNCHANGED)
        if idx <= per_training_n:
            cv2.imwrite(os.path.join(paths.data_training_normal, file), img)
            ct += 1
        else:
            cv2.imwrite(os.path.join(paths.data_validation_normal, file), img)
            cv += 1
        print(
            f"Normal frames: Training = {ct} Validation = {cv} T = {round(ct/lfn,4)} V = {round(cv/lfn,4)}", end="\r")
    print(
        f"Normal frames: Training = {ct} Validation = {cv} T = {round(ct/lfn,4)} V = {round(cv/lfn,4)}")
    # print("Normal frames: #Training = %i #Validation = %i" % (ct, cv))
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


def frames(aug=False, cant=-1):
    l_n, l_a = leertxt.leer(
        paths.normal_training_data_txt,
        paths.anomalous_training_data_txt
    )
    conta_a, conta_n = 0, 0
    nfn, nfa = 1,2#3, 6  # 3, 14  # 2, 14
    print("..................................................................................\nExtrayendo Frames Anomalous")
    for i, video in enumerate(l_a):
        if i + 1 > cant and cant != -1:
            break
        print("+++++++++++++++++++++++++++++++++++++\n", vars(video))
        conta_a, conta_n = extract.video_extract(
            video,
            os.path.join(paths.anomalous_data_set, video.name),
            i,
            conta_a,
            conta_n,
            nfa,
            nfn,
            aug
        )
    print("..................................................................................\nExtrayendo Frames Normales")
    for i, video in enumerate(l_n):
        if i + 1 > cant and cant != -1:
            break
        print("+++++++++++++++++++++++++++++++++++++\n", vars(video))
        conta_a, conta_n = extract.video_extract(
            video,
            os.path.join(paths.normal_data_set, video.name),
            i,
            conta_a,
            conta_n,
            nfa,
            nfn,
            aug
        )

    return conta_a, conta_n


def main():
    if len(sys.argv) == 1:
        canti = -1
        augmen = 1
    elif len(sys.argv) == 2:
        canti = -1
        augmen = int(sys.argv[1])
    else:
        canti = int(sys.argv[2])
        augmen = int(sys.argv[1])

    # Crear_Directorios_Training()
    c_a, c_n = frames(augmen, canti)
    print("Frames anomalos: ", c_a)
    print("Frames normales: ", c_n)
    print("Frames totales: ", c_n+c_a)
    # DataRandom()
    # python3 main.py augmentation cuantos
    # python3 main.py 1 2


if __name__ == '__main__':
    inicio = datetime.now()
    main()
    fin = datetime.now()
    print(inicio)
    print(fin - inicio)
    print(fin)
