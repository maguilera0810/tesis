from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
import cv2
import leertxt
import extract_texting
import pygame
import shutil as st
import rutas_data_preparation as rt
from pygame.locals import *
from tkinter import *

pygame.init()

cwd = ".."
PATHS = rt.Directorios(cwd=cwd)


WIDTH, HEIGHT = 224, 224

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800


IMG_SIZE_X, IMG_SIZE_Y = 700, 500
IMG_POS_X, IMG_POS_Y = 0, 0

TEXT_WIDTH, TEXT_HEIGHT = 400, 300
TEXT_POS_X, TEXT_POS_Y = SCREEN_WIDTH - TEXT_WIDTH - 100, 50

GF_WIDTH, GF_HEIGHT = 400, 400
GF_POS_X, GF_POS_Y = SCREEN_WIDTH - GF_WIDTH - 50, SCREEN_HEIGHT - 50

TB_WIDTH, TB_HEIGHT = 700, 120
TB_POS_X, TB_POS_Y = 50, SCREEN_HEIGHT - TB_HEIGHT - 50

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)

STYLE_1 = pygame.font.SysFont("Arial", 15)
STYLE_2 = pygame.font.SysFont("Arial", 20)


BUCLE = True

modelos = [
    os.path.join(PATHS.checkpoints_resnet, "ResNet50_model_weights.h5"),
    os.path.join(PATHS.checkpoints_inception, "InceptionV3_model_weights.h5"),
    os.path.join(PATHS.checkpoints_frankensnet, "FrankensNet_model_weights.h5")
]

v_name = ""


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

color2 = 'dark slate gray'
color1 = 'black'
blanco = "white"
margin_left = 50
margin_up = 50
desp_y = 20


def guindou():

    def click():
        global prueba1
        global prueba2
        global prueba3
        global prueba4
        global v_name
        prueba1 = bool(var1.get())
        prueba2 = bool(var2.get())
        prueba3 = bool(var3.get())
        prueba4 = bool(var.get())
        print("---------")
        if not prueba4:
            v_name = video_name.get()
            print(v_name)

        window.destroy()

    def terminar():
        global BUCLE
        BUCLE = False
        window.destroy()

    window = Tk()
    window.title("Anomaly Detection System")
    window_width = 500
    window_height = 300
    window.geometry(f"{window_width}x{window_height}")
    window.config(background=color1)
    window.protocol("WM_DELETE_WINDOW", terminar)
    Label(window, text="Models:", bg=color1, fg=blanco).place(
        x=margin_left, y=margin_up)
    var1 = IntVar()
    Checkbutton(window, text="ResNet50 \t", variable=var1, width=0).place(
        x=margin_left, y=margin_up+desp_y)
    var2 = IntVar()
    Checkbutton(window, text="InceptionV3\t", variable=var2, width=0).place(
        x=margin_left, y=margin_up+desp_y*2)
    var3 = IntVar()
    Checkbutton(window, text="FrankensNet\t", variable=var3, width=0).place(
        x=margin_left, y=margin_up+desp_y*3)

    Label(window, text="Video:", bg=color1, fg=blanco).place(
        x=margin_left+200, y=margin_up)
    var = IntVar()
    video_name = Entry(window, width=19, fg=color1)
    video_name.place(x=margin_left + 200, y=margin_up+desp_y)
    R1 = Radiobutton(window, text="Choose one\t", variable=var,
                     value=0)
    R1.place(x=margin_left + 200, y=margin_up+desp_y*2)
    R2 = Radiobutton(window, text="All videos(default)\t",
                     variable=var, value=1)
    R2.place(x=margin_left + 200, y=margin_up+desp_y*3)

    Button(window, text='Quit', command=terminar).place(
        x=margin_left, y=margin_up + 100)
    Button(window, text='Enviar', command=click).place(
        x=margin_left+200, y=margin_up + 100)

    window.mainloop()


def tabla(screen, pos_x=TB_POS_X, pos_y=TB_POS_Y, data_r=None, data_i=None, data_f=None, ancho=TB_WIDTH, alto=TB_HEIGHT, style=STYLE_1):
    posy = pos_y
    desp_x = ancho/5
    filas = 1 + (int(data_r != None)+int(data_i != None)+int(data_f != None))
    desp_y = 30
    cols = [" ", "TP", "FP", "TN", "FN", " "]
    models = []
    if data_r != None:
        models.append(["Resnet50", data_r])
    if data_i != None:
        models.append(["Inception", data_i])
    if data_f != None:
        models.append(["FrankenNet", data_f])

    for i in range(6):
        start_pos = pos_x + desp_x * i, pos_y
        end_pos = pos_x + desp_x * i, pos_y + filas * desp_y
        pygame.draw.line(screen, white, start_pos, end_pos, 1)
        Texto = style.render(cols[i], True, white)
        screen.blit(Texto, (start_pos[0]+desp_x//2-5, start_pos[1]+desp_y//2))
    idx = 0
    for i in range(filas+1):
        start_pos = pos_x, pos_y + i * desp_y
        end_pos = pos_x + ancho, pos_y + i * desp_y
        pygame.draw.line(screen, white, start_pos, end_pos, 1)
        # Texto = style.render("ResNet50", True, white)
        # print(i,end="\r")

        if i > 0 and idx < len(models):
            Texto = style.render(models[idx][0], True, white)
            screen.blit(Texto, (start_pos[0] + 20, start_pos[1]+desp_y//2))
            datos = models[idx][1]

            Texto = style.render(str(datos[0]), True, white)
            screen.blit(Texto, (start_pos[0] + 20 +
                                desp_x*1, start_pos[1]+desp_y//2))
            Texto = style.render(str(datos[1]), True, white)
            screen.blit(Texto, (start_pos[0] + 20 +
                                desp_x*2, start_pos[1]+desp_y//2))
            Texto = style.render(str(datos[2]), True, white)
            screen.blit(Texto, (start_pos[0] + 20 +
                                desp_x*3, start_pos[1]+desp_y//2))
            Texto = style.render(str(datos[3]), True, white)
            screen.blit(Texto, (start_pos[0] + 20 +
                                desp_x*4, start_pos[1]+desp_y//2))
            idx += 1


def texto_data(screen, pos_x=TEXT_POS_X, pos_y=TEXT_POS_Y, data_r=None, data_i=None, data_f=None, ancho=TEXT_WIDTH, alto=TEXT_HEIGHT, style=STYLE_1):
    posy = pos_y
    partes = 3*(int(data_r != None)+int(data_i != None)+int(data_f != None))
    desp = alto/partes
    # print(data_r, data_i, data_f,"-----------------------------------------------------------------------------")
    if data_r != None:
        Texto = style.render(f"Resnet:", True, white)
        screen.blit(Texto, (pos_x, posy))
        posy += desp
        Texto = style.render(
            f"Anomalous score = {data_r.get('anomalous')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp
        Texto = style.render(
            f"Normal Score = {data_r.get('normal')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp
    if data_i != None:
        Texto = style.render(f"Inception:", True, white)
        screen.blit(Texto, (pos_x, posy))
        posy += desp
        Texto = style.render(
            f"Anomalous score = {data_i.get('anomalous')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp
        Texto = style.render(
            f"Normal Score = {data_i.get('normal')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp
    if data_f != None:
        Texto = style.render(f"FrankensNet:", True, white)
        screen.blit(Texto, (pos_x, posy))
        posy += desp
        Texto = style.render(
            f"Anomalous score = {data_f.get('anomalous')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp
        Texto = style.render(
            f"Normal Score = {data_f.get('normal')}", True, white)
        screen.blit(Texto, (pos_x + 50, posy))
        posy += desp


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


def grafica(screen, pos_x=GF_POS_X, pos_y=GF_POS_Y, data_r=None, data_i=None, data_f=None, ancho=GF_WIDTH, alto=GF_HEIGHT, width=0, lines=10, style=STYLE_1):
    grosor = 50
    pos1 = pos_x
    espacio = (ancho-100*(int(data_i != None) +
                          int(data_r != None)+int(data_f != None)))/(int(data_i != None) +
                                                                     int(data_r != None)+int(data_f != None)+1)
    pos_x += espacio
    name_r, name_i, name_f = "ResNet50", "Inception_V3", "FrankensNet"
    if data_i != None:
        rect_a_i = rx, ry, rw, rh = pos_x, pos_y, grosor, - \
            data_i.get('anomalous')*alto
        pos_name_i = rx, ry+10
        pos_x += grosor
        rect_n_i = rx, ry2, rw, rh = pos_x, pos_y, grosor, - \
            data_i.get('normal')*alto
        pos_x += grosor + espacio
        # print("-----------------------------------------------inception-----------------------------------------------------*")
    if data_r != None:
        rect_a_r = rx, ry2, rw, rh = pos_x, pos_y, grosor, - \
            data_r.get('anomalous')*alto
        pos_name_r = rx, ry2+10
        pos_x += grosor
        rect_n_r = rx, ry2, rw, rh = pos_x, pos_y, grosor, - \
            data_r.get('normal')*alto
        pos_x += grosor + espacio
        # print("-----------------------------------------------resnet-----------------------------------------------------*")
    if data_f != None:
        rect_a_f = rx, ry2, rw, rh = pos_x, pos_y, grosor, - \
            data_f.get('anomalous')*alto
        pos_name_f = rx, ry2+10
        pos_x += grosor
        rect_n_f = rx, ry2, rw, rh = pos_x, pos_y, grosor, - \
            data_f.get('normal')*alto
        pos_x += grosor + espacio
        # print("-----------------------------------------------frankensnet-----------------------------------------------------*")

    for i in range(lines+1):
        start_pos = pos1, pos_y-alto*i/lines
        end_pos = pos_x, pos_y-alto*i/lines
        pygame.draw.line(screen, white, start_pos, end_pos, 1)
        Texto = style.render(f"{round(i/lines,2)}", True, white)
        screen.blit(Texto, (start_pos[0]-60, start_pos[1]))
    if data_i != None:
        pygame.draw.rect(screen, red, rect_a_i, width)
        pygame.draw.rect(screen, green, rect_n_i, width)
        Texto = style.render(
            name_i, True, white)
        screen.blit(Texto, pos_name_i)
    if data_r != None:
        pygame.draw.rect(screen, red, rect_a_r, width)
        pygame.draw.rect(screen, green, rect_n_r, width)
        Texto = style.render(
            name_r, True, white)
        screen.blit(Texto, pos_name_r)
    if data_f != None:
        pygame.draw.rect(screen, red, rect_a_f, width)
        pygame.draw.rect(screen, green, rect_n_f, width)
        Texto = style.render(
            name_f, True, white)
        screen.blit(Texto, pos_name_f)


fps = 30
f_n, f_a = leertxt.leer(
    PATHS.normal_training_data_txt,
    PATHS.anomalous_training_data_txt
)


def paigeim():
    global SCREEN_HEIGHT
    global SCREEN_WIDTH
    if prueba1:
        parameters_r = fp_r, fn_r, tp_r, tn_r = 0, 0, 0, 0
    else:
        parameters_r = None
    if prueba2:
        parameters_i = fp_i, fn_i, tp_i, tn_i = 0, 0, 0, 0
    else:
        parameters_i = None
    if prueba3:
        parameters_f = fp_f, fn_f, tp_f, tn_f = 0, 0, 0, 0
    else:
        parameters_f = None
    lr, li, lf, y_test, y_pred_r, y_pred_i, y_pred_f = [], [], [], [], [], [], []
    path = "screens"
    salir = False
    if prueba4:
        if True:
            for video in f_a:

                if salir:
                    screen.fill(black)
                    pygame.display.flip()
                    break
                else:
                    screen = pygame.display.set_mode(
                        (SCREEN_WIDTH, SCREEN_HEIGHT), RESIZABLE)
                    pygame.display.set_caption("Testing Anomalous Detection")
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

                    while True and not salir:  # fps._numFrames < 120

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:

                                salir = True

                        check, frame = video_capture.read()  # get current frame
                        # frameId = video_capture.get(1)  # current frame number
                        # print(check)
                        if check:
                            f_name = os.path.join(path, f"{i}_alpha.png")
                            # write frame image to file
                            cv2.imwrite(filename=f_name, img=frame)

                            if not extract_texting.checkear(video.tramos_no_usar, fps, i):

                                if i % 25 == 0:
                                    # test_image = image.load_img(f_name, target_size=(255, 255))

                                    test_image = image.load_img(
                                        f_name, target_size=(WIDTH, HEIGHT))
                                    test_image = image.img_to_array(test_image)
                                    test_image = np.expand_dims(
                                        test_image, axis=0)

                                    if prueba1:
                                        result_resnet = classifier_resnet.predict(
                                            test_image)
                                        data_resnet = {
                                            "anomalous": result_resnet[0][0], "normal": result_resnet[0][1]}
                                    else:
                                        data_resnet = None
                                    if prueba2:
                                        result_inception = classifier_inception.predict(
                                            test_image)
                                        data_inception = {
                                            "anomalous": result_inception[0][0], "normal": result_inception[0][1]}
                                    else:
                                        data_inception = None
                                    if prueba3:
                                        result_frankensnet = classifier_frankensnet.predict(
                                            test_image)
                                        data_frankensnet = {
                                            "anomalous": result_frankensnet[0][0], "normal": result_frankensnet[0][1]}
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

                                    # parameters_i  = [tp_i, fp_i, tn_i, fn_i]
                                    screen.fill(black)

                                    tabla(screen=screen, pos_x=TB_POS_X, pos_y=TB_POS_Y, data_r=parameters_r,
                                          data_i=parameters_i, data_f=parameters_f, ancho=TB_WIDTH, alto=TB_HEIGHT, style=STYLE_1)
                                    texto_data(screen=screen,  pos_x=TEXT_POS_X, pos_y=TEXT_POS_Y, data_i=data_inception,
                                               data_r=data_resnet, data_f=data_frankensnet, ancho=TEXT_WIDTH, alto=TEXT_HEIGHT, style=STYLE_2)
                                    grafica(screen=screen,  pos_x=GF_POS_X, pos_y=GF_POS_Y, data_i=data_inception,
                                            data_r=data_resnet, data_f=data_frankensnet, ancho=GF_WIDTH, alto=GF_HEIGHT, width=0, lines=20, style=STYLE_1)
                                if prueba1:
                                    parameters_r = tp_r, fp_r, tn_r, fn_r
                                    res_r = f"fp: {fp_r} fn: {fn_r} tp: {tp_r} tn: {tn_r} total: {fp_r + fn_r + tp_r + tn_r} -{parameters(tp_r, fp_r, tn_r, fn_r)}-| "
                                else:
                                    res_r = ""
                                if prueba2:
                                    parameters_i = tp_i, fp_i, tn_i, fn_i
                                    res_i = f"fp: {fp_i} fn: {fn_i} tp: {tp_i} tn: {tn_i} total: {fp_i + fn_i + tp_i + tn_i} -{parameters(tp_i, fp_i, tn_i, fn_i)}-|"
                                else:
                                    res_i = ""
                                if prueba3:
                                    parameters_f = tp_f, fp_f, tn_f, fn_f
                                    res_f = f"fp: {fp_f} fn: {fn_f} tp: {tp_f} tn: {tn_f} total: {fp_f + fn_f + tp_f + tn_f} -{parameters(tp_f, fp_f, tn_f, fn_f)}-| "
                                else:
                                    res_f = ""
                                res = res_r + res_i + res_f                                     
                                # res = f"fp: {fp_r} fn: {fn_r} tp: {tp_r} tn: {tn_r} total: {fp_r + fn_r + tp_r + tn_r}  \t{parameters(tp_r, fp_r, tn_r, fn_r)}"
                                print(res, end="\r")
                                try:
                                    imagen = pygame.image.load(f_name)
                                    imagen = pygame.transform.scale(
                                        imagen, (IMG_SIZE_X, IMG_SIZE_Y))
                                    screen.blit(imagen, (IMG_POS_X, IMG_POS_Y))
                                except:
                                    print(
                                        "ERROR PYGAME ANOMALOUS+++++++++++++++++++++++++++++++++++")
                                    break
                                pygame.display.flip()  # -------------------------------------
                            i += 1
                        else:
                            break
                    video_capture.release()

        else:
            if not salir:
                for video in f_n:
                    if salir:
                        screen.fill(black)
                        pygame.display.flip()
                        break
                    else:
                        screen = pygame.display.set_mode(
                            (SCREEN_WIDTH, SCREEN_HEIGHT), RESIZABLE)
                        pygame.display.set_caption(
                            "Testing Anomalous Detection")
                        print(vars(video))
                        try:
                            os.mkdir(path)
                        except:
                            st.rmtree(path)
                            os.mkdir(path)
                        try:
                            video_capture = cv2.VideoCapture(
                                os.path.join(PATHS.normal_data_set, video.name))
                            # print(os.path.join(PATHS.anomalous_data_set, video.name))
                        except:
                            print(
                                "ERROR CV2 ANOMALOUS-----------------------------------")
                            continue
                        i = 0
                        while True and not salir:  # fps._numFrames < 120
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:

                                    salir = True

                            check, frame = video_capture.read()  # get current frame
                            # frameId = video_capture.get(1)  # current frame number
                            # print(check)
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
                                    else:
                                        data_resnet = None
                                    if prueba2:
                                        result_inception = classifier_inception.predict(
                                            test_image)
                                        data_inception = {
                                            "anomalous": result_inception[0][0], "normal": result_inception[0][1]}
                                        if data_inception.get('anomalous') > data_inception.get('normal'):
                                            fp_i += 1
                                        else:
                                            tn_i += 1
                                    else:
                                        data_inception = None
                                    if prueba3:
                                        result_frankensnet = classifier_frankensnet.predict(
                                            test_image)
                                        data_frankensnet = {
                                            "anomalous": result_frankensnet[0][0], "normal": result_frankensnet[0][1]}
                                        if data_frankensnet.get('anomalous') > data_frankensnet.get('normal'):
                                            fp_f += 1
                                        else:
                                            tn_f += 1
                                    else:
                                        data_frankensnet = None
                                    screen.fill(black)
                                    tabla(screen=screen, pos_x=TB_POS_X, pos_y=TB_POS_Y, data_r=parameters_r,
                                          data_i=parameters_i, data_f=parameters_f, ancho=TB_WIDTH, alto=TB_HEIGHT, style=STYLE_1)
                                    texto_data(screen=screen,  pos_x=TEXT_POS_X, pos_y=TEXT_POS_Y, data_i=data_inception,
                                               data_r=data_resnet, data_f=data_frankensnet, ancho=TEXT_WIDTH, alto=TEXT_HEIGHT, style=STYLE_2)
                                    grafica(screen=screen,  pos_x=GF_POS_X, pos_y=GF_POS_Y, data_i=data_inception, data_r=data_resnet,
                                            data_f=data_frankensnet, ancho=GF_WIDTH, alto=GF_HEIGHT, width=0, lines=20, style=STYLE_1)
                                    if prueba1:
                                        parameters_r = tp_r, fp_r, tn_r, fn_r
                                    if prueba2:
                                        parameters_i = tp_i, fp_i, tn_i, fn_i
                                    if prueba3:
                                        parameters_f = tp_f, fp_f, tn_f, fn_f
                                    #res_i = f"fp: {fp_i} fn: {fn_i} tp: {tp_i} tn: {tn_i} total: {fp_i + fn_i + tp_i + tn_i} | "
                                    res_r = f"fp: {fp_r} fn: {fn_r} tp: {tp_r} tn: {tn_r} total: {fp_r + fn_r + tp_r + tn_r} | "
                                    print(res_i, end="\r")
                                    try:
                                        imagen = pygame.image.load(f_name)
                                        imagen = pygame.transform.scale(
                                            imagen, (IMG_SIZE_X, IMG_SIZE_Y))
                                        screen.blit(
                                            imagen, (IMG_POS_X, IMG_POS_Y))
                                    except:
                                        print(
                                            "ERROR PYGAME NORMAL+++++++++++++++++++++++++++++++++++")
                                        break
                                pygame.display.flip()  # -------------------------------------

                                i += 1
                            else:
                                break
                    video_capture.release()

    else:

        screen = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), RESIZABLE)
        pygame.display.set_caption("Testing Anomalous Detection")

        try:
            os.mkdir(path)
        except:
            st.rmtree(path)
            os.mkdir(path)
        try:
            global v_name
            video_capture = cv2.VideoCapture(v_name)
            # print(os.path.join(PATHS.anomalous_data_set, video.name))
        except:
            print("ERROR CV2 ANOMALOUS-----------------------------------")
            #salir = True
        i = 0

        while True and not salir:  # fps._numFrames < 120

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    screen.fill(black)
                    pygame.display.flip()
                    salir = True
                    break
            if salir:
                break

            check, frame = video_capture.read()  # get current frame
            if i == 0 and not check:
                print("No existe el video")
            if check:
                f_name = os.path.join(path, f"{i}_alpha.png")
                # write frame image to file
                cv2.imwrite(filename=f_name, img=frame)

                if i % 25 == 0:
                        # test_image = image.load_img(f_name, target_size=(255, 255))

                    test_image = image.load_img(
                        f_name, target_size=(WIDTH, HEIGHT))
                    # "84985.jpg", target_size=(WIDTH, HEIGHT))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis=0)

                    if prueba1:
                        result_resnet = classifier_resnet.predict(
                            test_image)
                        data_resnet = {
                            "anomalous": result_resnet[0][0], "normal": result_resnet[0][1]}
                    else:
                        data_resnet = None
                    if prueba2:
                        result_inception = classifier_inception.predict(
                            test_image)
                        data_inception = {
                            "anomalous": result_inception[0][0], "normal": result_inception[0][1]}
                    else:
                        data_inception = None
                    if prueba3:
                        result_frankensnet = classifier.predict(
                            test_image)
                        data_frankensnet = {
                            "anomalous": result_frankensnet[0][0], "normal": result_frankensnet[0][1]}
                    else:
                        data_frankensnet = None
                    screen.fill(black)
                    texto_data(screen=screen,  pos_x=TEXT_POS_X, pos_y=TEXT_POS_Y, data_i=data_inception,
                               data_r=data_resnet, data_f=data_frankensnet, ancho=TEXT_WIDTH, alto=TEXT_HEIGHT, style=STYLE_2)
                    grafica(screen=screen,  pos_x=GF_POS_X, pos_y=GF_POS_Y, data_i=data_inception,
                            data_r=data_resnet, data_f=data_frankensnet, ancho=GF_WIDTH, alto=GF_HEIGHT, width=0, lines=20, style=STYLE_1)

                    try:
                        imagen = pygame.image.load(f_name)
                        imagen = pygame.transform.scale(
                            imagen, (IMG_SIZE_X, IMG_SIZE_Y))
                        screen.blit(imagen, (IMG_POS_X, IMG_POS_Y))
                    except:
                        print(
                            "ERROR PYGAME ANOMALOUS+++++++++++++++++++++++++++++++++++")
                        break
                    pygame.display.flip()  # -------------------------------------
                i += 1
            else:
                break
        video_capture.release()


def main():

    while BUCLE:
        print("-------------DE NUEVO----------------")
        guindou()
        if BUCLE:
            paigeim()


main()
# ------------------------------------------------------------------------------------------------------
