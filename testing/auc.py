#from keras.models import load_model
#from keras.preprocessing import image
#import numpy as np
import sys
import os
import cv2
import leertxt
import extract_texting
import shutil as st
import rutas_data_preparation as rt
from datetime import datetime
from tkinter import *
from plots import Plot
#import h5py
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_test =       [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]

y_pred_keras = [0.98,1, 0,0, 0.9,0.96, 0.1,0, 0,1, 1,0, 1,1, 1,0 ,0,1  ]

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)