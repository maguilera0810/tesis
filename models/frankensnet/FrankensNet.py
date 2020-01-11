from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, Input, Concatenate, Reshape
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, BaseLogger, Callback, LambdaCallback
from matplotlib import pyplot as plt
import glob
import numpy as np
import cv2
from datetime import datetime
import os
import rutas_data_preparation as rt
import warnings  # ---
import tensorflow as tf
from callbacks_mau import BatchData
from models_mau import FrankensNet

from plots import Plot


inicio = datetime.now()
cwd = ".."
paths = rt.Directorios(os.path.join(cwd, cwd))

print("//////////////////////////----I N I C I O -----///   ", inicio)

TRAIN_DIR = paths.data_training
VALIDATION_DIR = paths.data_validation
BATCH_SIZE = 100  # 25
HEIGHT = 224
WIDTH = 224
NUM_EPOCHS = 3  # 5
class_list = ["anomalous", "normal"]
# FC_LAYERS = [1024, 1024]
FC_LAYERS = [2048, 1024]  # cambio-------------------#
dropout = 0.5
LEARNING_RATE = 0.0001  # 0.000001 #0.00001
# num_train_images = 45215
# num_validation_images = 5023
if False:
    num_train_images = len([file for file in os.listdir(paths.data_training_normal)]) + \
        len([file for file in os.listdir(paths.data_training_normal)])
    num_validation_images = len([file for file in os.listdir(paths.data_validation_anomalous)]) + \
        len([file for file in os.listdir(paths.data_validation_normal)])

    print("steps train ", num_train_images, num_train_images // BATCH_SIZE)
    print("steps val ", num_validation_images, num_validation_images // BATCH_SIZE)


    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True
    )

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=(
                                                            HEIGHT, WIDTH),
                                                        batch_size=BATCH_SIZE)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                target_size=(
                                                                    HEIGHT, WIDTH),
                                                                batch_size=BATCH_SIZE)


frankens_model = FrankensNet(
    blocks=[6, 12, 24, 16], input_shape=(HEIGHT, WIDTH, 3), classes=2)


adam = Adam(lr=LEARNING_RATE)  # adam = Adam(lr=0.00001)
frankens_model.compile(
    adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath = "./checkpoints/" + "MobileInception" + "_model_weights.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor="acc", verbose=1, mode='max', save_best_only=True)

filepath_batch = paths.batch_data
filepath_epoch = paths.epoch_data
batchdata = BatchData(filepath_batch, filepath_epoch, "mobileinception")

callbacks_list = [
    batchdata,
    checkpoint

]


""" history = frankens_model.fit_generator(
    train_generator,
    epochs=NUM_EPOCHS,
    workers=8,
    steps_per_epoch=num_train_images // BATCH_SIZE,
    shuffle=True, callbacks=callbacks_list,
    validation_data=validation_generator,
    validation_steps=num_validation_images//BATCH_SIZE
) """

""" 
fin = datetime.now()
print(history.history)

print("//////////////////////////----I N I C I O -----///   ", inicio)
print("Duracion:  ", fin - inicio)
print("//////////////////////////------- F I N -------///   ", fin)
print("steps train ", num_train_images // BATCH_SIZE)
print("steps val ", num_validation_images // BATCH_SIZE)
# Plot the training and validation loss + accuracy


def GuardarEpocas(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(filepath_epoch)
    file = open(f"{filepath_epoch}/{model_name}_{datetime.now()}.txt", "w")
    for a, va, l, vl in zip(acc, val_acc, loss, val_loss):
        print(a, va, l, vl)
        file.write((str(a)+"\t"+str(va)+"\t"+str(l)+"\t"+str(vl)+"\n"))
    file.close()


print("////////////////////////+++++++++++++++++++")
GuardarEpocas(history, "resnet")
print("////////////////////////-------------------")
Plot(history) """
