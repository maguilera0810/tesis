import os
import shutil as st
import cv2
import numpy as np

def checkear(tramos, fps, frame):
    res = False
    i = 0
    while i < len(tramos) and not res:
        res = res or (int(tramos[i].inicio)*fps <= frame <=
                      (int(tramos[i].inicio) + int(tramos[i].duracion))*fps)
        i += 1
    return res
