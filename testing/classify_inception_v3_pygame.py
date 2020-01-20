import tensorflow as tf
import sys
import os
import cv2
import math
import leertxt

import rutas_testing as rt
import extract_texting

import shutil as st

import pygame
from pygame.locals import *

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# constants----------------------

SCREEN_WIDTH = 1250
SCREEN_HEIGHT = 550+50

img_pos_x = 0
img_pos_y = 0
img_size_x = 700
img_size_y = 500

text_pos_x = img_pos_x + img_size_x + 30
text_pos_y = 50

gf_pos_x = text_pos_x + 100
gf_pos_y = img_size_y

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
tf_files = rt.tf_files_inception_v3_model

label_lines = [line.rstrip() for line
               in tf.gfile.GFile(os.path.join(tf_files, "retrained_labels.txt"))]

with tf.gfile.FastGFile(os.path.join(tf_files, "retrained_graph.pb"), 'rb') as f:

    graph_def = tf.GraphDef()  # The graph-graph_def is a saved copy of a TensorFlow graph;
    # Parse serialized protocol buffer data into variable
    graph_def.ParseFromString(f.read())
    # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor
    _ = tf.import_graph_def(graph_def, name='')

# video_path = sys.argv[1]
# writer = None
paigeim = int(sys.argv[1])
# classify.py for video processing.
# This is the interesting part where we actually changed the code:
#############################################################


def grafica(screen, data, pos_x, pos_y, franja):
    ancho = 100
    pos1 = pos_x
    pos_x += 50
    factor = 300
    if data[0][0] == "anomalous":

        texto1, texto2, data1, data2 = data[0][0], data[1][0], data[0][1], data[1][1]
        rect1 = rx1, ry1, rw1, rh1 = pos_x, pos_y, ancho, -data[0][1]*factor
        pos_x += ancho+75
        rect2 = rx2, ry2, rw2, rh2 = pos_x, pos_y, ancho, -data[1][1]*factor
    else:
        texto1, texto2, data1, data2 = data[1][0], data[0][0], data[1][1], data[0][1]
        rect1 = rx1, ry1, rw1, rh1 = pos_x, pos_y, ancho, -data[1][1]*factor
        pos_x += ancho + 75
        rect2 = rx2, ry2, rw2, rh2 = pos_x, pos_y, ancho, -data[0][1]*factor
    start_pos1 = pos1, pos_y-factor
    end_pos1 = pos_x+ancho+50, pos_y-factor
    start_pos2 = pos1, pos_y
    end_pos2 = pos_x+ancho+50, pos_y
    start_pos1 = pos1, pos_y-factor
    end_pos1 = pos_x+ancho+50, pos_y-factor
    start_pos2 = pos1, pos_y
    end_pos2 = pos_x+ancho+50, pos_y
    start_pos3 = pos1, pos_y - factor*franja
    end_pos3 = pos_x+ancho+50, pos_y - factor*franja
    pygame.draw.line(screen, white, start_pos3, end_pos3, 1)
    Texto = style_1.render(
        'franja', True, white)
    screen.blit(Texto, (start_pos3[0]-60, start_pos3[1]))

    pygame.draw.line(screen, white, start_pos1, end_pos1, 1)
    Texto = style_1.render(
        '100%', True, white)
    screen.blit(Texto, (start_pos1[0]-60, start_pos1[1]))
    pygame.draw.line(screen, white, start_pos2, end_pos2, 1)
    Texto = style_1.render(
        '  0%', True, white)
    screen.blit(Texto, (start_pos2[0]-60, start_pos2[1]))
    width = 0
    pygame.draw.rect(screen, red, rect1, width)
    pygame.draw.rect(screen, green, rect2, width)
    Texto = style_1.render(
        texto1, True, white)
    screen.blit(Texto, (rx1, ry1+10))
    Texto = style_1.render(
        texto2, True, white)
    screen.blit(Texto, (rx2, ry2+10))


path = "screens"

with tf.Session() as sess:
    fps = 30
    f_n, f_a = leertxt.leer(
        rt.n_ts_data_txt,
        rt.a_ts_data_txt
    )
    print(len(f_n), len(f_a))
    fp, fn, tp, tn, na = 0, 0, 0, 0, 0

    if paigeim:
        pygame.init()
        style_1 = pygame.font.SysFont("Arial", 30)
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Testing Anomalous Detection - GoogLeNet")
        if False:    
            for video in f_a:
                print(vars(video))
                try:
                    os.mkdir(path)
                except:
                    st.rmtree(path)
                    os.mkdir(path)
                try:
                    video_capture = cv2.VideoCapture(
                        os.path.join(rt.anomalous_data_set, video.name))
                    i = 0
                    while True:  # fps._numFrames < 120

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sys.exit()
                        
                        frame = video_capture.read()[1]  # get current frame
                        frameId = video_capture.get(1)  # current frame number
                        f_name = os.path.join(
                            path, str(i)+"alpha.png")
                        cv2.imwrite(filename=f_name,
                                    img=frame)  # write frame image to file
                        print(f_name)
                        if not extract_texting.checkear(video.tramos_no_usar, fps, i):
                            if i % 10 == 0:
                                image_data = tf.gfile.FastGFile(
                                    f_name, 'rb').read()  # get this image file
                                softmax_tensor = sess.graph.get_tensor_by_name(
                                    'final_result:0')
                                predictions = sess.run(softmax_tensor,
                                                    {'DecodeJpeg/contents:0': image_data})     # analyse the image

                                screen.fill([0, 0, 0])

                                top_k = predictions[0].argsort(
                                )[-len(predictions[0]):][::-1]
                                # top_k = predictions[0][-len(predictions[0]):][::-1]
                                pos = 0
                                data = []
                                for node_id in top_k:
                                    datos = human_string, score = label_lines[node_id], predictions[0][node_id]
                                    data.append(datos)
                                    # cv2.putText(frame, '%s (score = %.5f)' % (human_string, score),(40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

                                    # print('%s (score = %.5f)' % (human_string, score))
                                    pos = pos + 1
                                if data[0][0] == "normal":
                                    aux = data[0]
                                    data[0] = data[1]
                                    data[1] = aux
                                Texto = style_1.render(
                                    '%s (score = %.5f)' % (data[0][0], data[0][1]), True, white)
                                screen.blit(Texto, (text_pos_x, text_pos_y))
                                Texto = style_1.render(
                                    '%s (score = %.5f)' % (data[1][0], data[1][1]), True, white)
                                screen.blit(Texto, (text_pos_x, text_pos_y+30))
                                Texto = style_1.render(
                                    'frame = %d' % i, True, white)
                                screen.blit(Texto, (text_pos_x, text_pos_y+pos*30))
                                grafica(screen, data, gf_pos_x, gf_pos_y, 0.60)
                                if data[0][1] >= 0.60:
                                    pygame.draw.rect(
                                        screen, red, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                                    if extract_texting.checkear(video.tramosAnomalos, fps, i) and not extract_texting.checkear(video.tramos_no_usar, fps, i):
                                        tp += 1
                                    else:
                                        fp += 1
                                elif data[1][1] >= 0.60:
                                    pygame.draw.rect(
                                        screen, green, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                                    if extract_texting.checkear(video.tramosAnomalos, fps, i) and not extract_texting.checkear(video.tramos_no_usar, fps, i):
                                        fn += 1
                                    else:
                                        tn += 1
                                else:
                                    pygame.draw.rect(
                                        screen, blue, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                                    na += 1
                            # print("\n")
                            res = 'fp: %i fn: %i tp: %i tn: %i na: %i total: %i' % (
                                fp, fn, tp, tn, na, fp + fn + tp + tn + na)
                            print(res, end="\r")
                            try:
                                imagen = pygame.image.load(f_name)
                                imagen = pygame.transform.scale(
                                    imagen, (img_size_x, img_size_y))
                                screen.blit(imagen, (img_pos_x, img_pos_y))
                            except:
                                print(
                                    "ERROR PYGAME ANOMALOUS+++++++++++++++++++++++++++++++++++")
                                break
                            pygame.display.flip()  # -------------------------------------
                        """ if writer is None:
                            # initialize our video writer
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            writer = cv2.VideoWriter("recognized.avi", fourcc, 30,
                                                    (frame.shape[1], frame.shape[0]), True) """
                        i = i + 1
                        # write the output frame to disk
                        # writer.write(frame)

                        """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('image', 900, 900)
                        cv2.imshow("image", frame)  # show frame in window
                        cv2.waitKey(1)  # wait 1ms -> 0 until key input """
                    # writer.release()
                    video_capture.release()
                    # cv2.destroyAllWindows()
                except:
                    print("ERROR CV2 ANOMALOUS-----------------------------------")

        for video in f_n:
            print(vars(video))
            try:
                os.mkdir(path)
            except:
                st.rmtree(path)
                os.mkdir(path)
            # try:
            video_capture = cv2.VideoCapture(
                os.path.join(rt.normal_data_set, video.name))
            i = 0
            while True:  # fps._numFrames < 120

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()

                frame = video_capture.read()[1]  # get current frame
                frameId = video_capture.get(1)  # current frame number
                f_name = os.path.join(
                    path, str(i)+"alpha.png")
                cv2.imwrite(filename=f_name,
                            img=frame)  # write frame image to file
                print(f_name)
                if i % 10 == 0:

                    image_data = tf.gfile.FastGFile(
                        f_name, 'rb').read()  # get this image file
                    softmax_tensor = sess.graph.get_tensor_by_name(
                        'final_result:0')
                    print("////////////////////////////////no  eror ////////////////////////////////////",i)
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})     # analyse the image
                    print("////////////////////////////////eror ////////////////////////////////////",i)

                    screen.fill([0, 0, 0])



                    top_k = predictions[0].argsort(
                    )[-len(predictions[0]):][::-1]
                    # top_k = predictions[0][-len(predictions[0]):][::-1]
                    pos = 0
                    data = []
                    for node_id in top_k:
                        datos = human_string, score = label_lines[node_id], predictions[0][node_id]
                        data.append(datos)
                        # cv2.putText(frame, '%s (score = %.5f)' % (human_string, score),(40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

                        # print('%s (score = %.5f)' % (human_string, score))
                        pos = pos + 1
                    if data[0][0] == "normal":
                        aux = data[0]
                        data[0] = data[1]
                        data[1] = aux
                    Texto = style_1.render(
                        '%s (score = %.5f)' % (data[0][0], data[0][1]), True, white)
                    screen.blit(Texto, (text_pos_x, text_pos_y))
                    Texto = style_1.render(
                        '%s (score = %.5f)' % (data[1][0], data[1][1]), True, white)
                    screen.blit(Texto, (text_pos_x, text_pos_y+30))
                    Texto = style_1.render(
                        'frame = %d' % i, True, white)
                    screen.blit(Texto, (text_pos_x, text_pos_y+pos*30))
                    grafica(screen, data, gf_pos_x, gf_pos_y, 0.60)
                    if data[0][1] >= 0.60:
                        pygame.draw.rect(
                            screen, red, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                        fp += 1
                    elif data[1][1] >= 0.60:
                        pygame.draw.rect(
                            screen, green, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                        tn += 1
                    else:
                        pygame.draw.rect(
                            screen, blue, (0, img_size_y+40, SCREEN_WIDTH, 60), 0)
                        na += 1
                # print("\n")
                res = 'fp: %i fn: %i tp: %i tn: %i na: %i total: %i' % (
                    fp, fn, tp, tn, na, fp + fn + tp + tn + na)
                print(res, end="\r")
                try:
                    imagen = pygame.image.load(f_name)
                    imagen = pygame.transform.scale(
                        imagen, (img_size_x, img_size_y))
                    screen.blit(imagen, (img_pos_x, img_pos_y))
                except:
                    print(
                        "ERROR PYGAME NORMAL+++++++++++++++++++++++++++++++++++")
                    break
                pygame.display.flip()  # -------------------------------------
                """ if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter("recognized.avi", fourcc, 30,
                                            (frame.shape[1], frame.shape[0]), True) """
                i = i + 1
                # write the output frame to disk
                # writer.write(frame)

                """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 900, 900)
                cv2.imshow("image", frame)  # show frame in window
                cv2.waitKey(1)  # wait 1ms -> 0 until key input """
            # writer.release()
            video_capture.release()
            # cv2.destroyAllWindows()
            # except:
            #print("ERROR CV2 NORMALES-----------------------------------")
    else:
        for video in f_a:
            print(vars(video))
            try:
                video_capture = cv2.VideoCapture(
                    os.path.join(rt.anomalous_data_set, video.name))
                i = 0
                while True:  # fps._numFrames < 120

                    if i % 10 == 0:
                        frame = video_capture.read()[1]  # get current frame
                        frameId = video_capture.get(1)  # current frame number
                        f_name = os.path.join(
                            rt.inception_v3_model, "screens", str(i)+"alpha.png")
                        cv2.imwrite(filename=f_name,
                                    img=frame)  # write frame image to file
                        image_data = tf.gfile.FastGFile(
                            f_name, 'rb').read()  # get this image file
                        softmax_tensor = sess.graph.get_tensor_by_name(
                            'final_result:0')
                        predictions = sess.run(softmax_tensor,
                                               {'DecodeJpeg/contents:0': image_data})     # analyse the image

                        top_k = predictions[0].argsort(
                        )[-len(predictions[0]):][::-1]
                        # top_k = predictions[0][-len(predictions[0]):][::-1]
                        pos = 0
                        data = []
                        for node_id in top_k:
                            datos = human_string, score = label_lines[node_id], predictions[0][node_id]
                            data.append(datos)
                            # cv2.putText(frame, '%s (score = %.5f)' % (human_string, score),(40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

                            # print('%s (score = %.5f)' % (human_string, score))
                            pos = pos + 1
                        if data[0][0] == "normal":
                            aux = data[0]
                            data[0] = data[1]
                            data[1] = aux

                        if data[0][1] >= 0.60:
                            if fps * int(video.start) <= i <= fps * (int(video.start)+int(video.end)):
                                tp += 1
                            else:
                                fp += 1
                        elif data[1][1] >= 0.60:
                            if fps * int(video.start) <= i <= fps * (int(video.start)+int(video.end)):
                                fn += 1
                            else:
                                tn += 1
                        else:
                            na += 1
                    # print("\n")
                    res = 'fp: %i fn: %i tp: %i tn: %i na: %i total: %i' % (
                        fp, fn, tp, tn, na, fp + fn + tp + tn + na)
                    print(res, end="\r")

                    i = i + 1
                    # write the output frame to disk
                    # writer.write(frame)

                    """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('image', 900, 900)
                    cv2.imshow("image", frame)  # show frame in window
                    cv2.waitKey(1)  # wait 1ms -> 0 until key input """
                # writer.release()
                video_capture.release()
                # cv2.destroyAllWindows()
            except:
                print("ERROR CV2 ANOMALOUS-----------------------------------")
        for video in f_n:
            print(vars(video))
            try:
                video_capture = cv2.VideoCapture(
                    os.path.join(rt.normal_data_set, video.name))
                i = 0
                while True:  # fps._numFrames < 120

                    if i % 10 == 0:
                        frame = video_capture.read()[1]  # get current frame
                        frameId = video_capture.get(1)  # current frame number
                        f_name = os.path.join(
                            rt.inception_v3_model, "screens", str(i)+"alpha.png")
                        cv2.imwrite(filename=f_name,
                                    img=frame)  # write frame image to file
                        image_data = tf.gfile.FastGFile(
                            f_name, 'rb').read()  # get this image file
                        softmax_tensor = sess.graph.get_tensor_by_name(
                            'final_result:0')
                        predictions = sess.run(softmax_tensor,
                                               {'DecodeJpeg/contents:0': image_data})     # analyse the image

                        top_k = predictions[0].argsort(
                        )[-len(predictions[0]):][::-1]
                        # top_k = predictions[0][-len(predictions[0]):][::-1]
                        pos = 0
                        data = []
                        for node_id in top_k:
                            datos = human_string, score = label_lines[node_id], predictions[0][node_id]
                            data.append(datos)
                            # cv2.putText(frame, '%s (score = %.5f)' % (human_string, score),(40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

                            # print('%s (score = %.5f)' % (human_string, score))
                            pos = pos + 1
                        if data[0][0] == "normal":
                            aux = data[0]
                            data[0] = data[1]
                            data[1] = aux
                        if data[0][1] >= 0.60:

                            fp += 1
                        elif data[1][1] >= 0.60:

                            tn += 1
                        else:

                            na += 1
                    # print("\n")
                    res = 'fp: %i fn: %i tp: %i tn: %i na: %i total: %i' % (
                        fp, fn, tp, tn, na, fp + fn + tp + tn + na)
                    print(res, end="\r")

                    i = i + 1
                    # write the output frame to disk
                    # writer.write(frame)

                    """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('image', 900, 900)
                    cv2.imshow("image", frame)  # show frame in window
                    cv2.waitKey(1)  # wait 1ms -> 0 until key input """
                # writer.release()
                video_capture.release()
                # cv2.destroyAllWindows()
            except:
                print("ERROR CV2 NORMALES-----------------------------------")

    res = 'fp: %i fn: %i tp: %i tn: %i na: %i' % (fp, fn, tp, tn, na)
    print(res)
    print('fp = %.5f' % (fp/(fp+tn)))
    print('fn = %.5f' % (fn/(fn+tp)))
    print('tp = %.5f' % (tp/(tp+fp)))
    print('tn = %.5f' % (tn/(tn+fn)))
    print('total frames = ', fp+fn+tp+tn+na)
