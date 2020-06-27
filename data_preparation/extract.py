import os
import shutil as st
import cv2
import numpy as np
import DataAugmentation as daug
import rutas_data_preparation as rt

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 500

cwd = ".."

paths = rt.Directorios(cwd=cwd)


def checkear(tramos, fps, frame):
    res = False
    i = 0
    while i < len(tramos) and not res:
        res = res or (int(tramos[i].inicio)*fps <= frame <=
                      (int(tramos[i].inicio) + int(tramos[i].duracion))*fps)
        i += 1
    return res


def video_extract(video, src, num_vid=0, conta_a=0, conta_n=0, n_frames_a=6, n_frames_n=4, aug=False):

    frame = 0
    n_frames_n, n_frames_a = 30//n_frames_n, 30//n_frames_a
    fps = 30

    cap = cv2.VideoCapture(src)
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if bool(video.type):
        # print('Total Frame Count:', length )
        while True and frame<5:
            check, img = cap.read()
            res = 'Tipo: %i Video: %i Processed: %i Img_N: %i Img_A: %i' % (
                0, num_vid, frame, conta_n, conta_a)
            if check:
                if not checkear(video.tramos_no_usar, fps, frame):
                    if checkear(video.tramosAnomalos, fps, frame) and False:
                        if frame % n_frames_a == 0:
                            # img = cv2.resize(img, (1920 // factor, 1080 // factor))

                            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            pat = os.path.join("temporal", str(conta_n) + "_original.jpg")
                            cv2.imwrite(pat, img)
                            conta_a+= 1
                            if aug:
                                conta_a += daug.img_aug(
                                    pat,
                                    img,
                                    "temporal",
                                    conta_a
                                )
                            #else:
                            t = 1
                            res = 'Tipo: %i Video: %i Processed: %i Img_N: %i Img_A: %i' % (
                                t, num_vid, frame, conta_n, conta_a)
                            print(res, end="\r")
                    else:
                        if frame % n_frames_n == 0:
                            # img = cv2.resize(img, (1920 // factor, 1080 // factor))
                            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            pat = os.path.join(
                                "temporal", str(conta_n) + "_original.jpg")
                            cv2.imwrite(pat, img)
                            conta_n += 1
                            if aug:
                                conta_n += daug.img_aug(
                                    pat,
                                    img,
                                    "temporal",
                                    conta_n
                                )
                            #else:
                                
                            t = 0
                            res = 'Tipo: %i Video: %i Processed: %i Img_N: %i Img_A: %i' % (
                                t, num_vid, frame, conta_n, conta_a)
                            print(res, end="\r")
                frame += 1

            else:
                print(res)
                break
    elif False:
        while True:
            check, img = cap.read()
            res = 'Tipo: %i Video: %i Processed: %i Img_N: %i Img_A: %i' % (
                0, num_vid, frame, conta_n, conta_a)
            if check:
                # if fps*ini <= frame <= fps*fin and frame > 0:
                if frame % n_frames_n == 0:
                    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    pat = os.path.join(
                        paths.data_temporal_normal, str(conta_n) + ".jpg")
                    if aug:
                        conta_n += daug.img_aug(
                            pat,
                            img,
                            paths.data_temporal_normal,
                            conta_n
                        )
                    else:
                        cv2.imwrite(pat, img)
                        conta_n += 1

                    t = 0
                    res = 'Tipo: %i Video: %i Processed: %i Img_N: %i Img_A: %i' % (
                        t, num_vid, frame, conta_n, conta_a)
                    print(res, end="\r")
                frame += 1
            else:
                print(res)
                break
    cap.release()
    return conta_a, conta_n


""" if __name__ == '__main__':
    video_extract('p2.avi', 2, 8) """
