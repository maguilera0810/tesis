from keras.applications.densenet import preprocess_input
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
BATCH_SIZE = 200  # 100
HEIGHT = 224
WIDTH = 224
NUM_EPOCHS = 5 # 5
class_list = ["anomalous", "normal"]
# FC_LAYERS = [1024, 1024]
FC_LAYERS = [1024]#[2048, 1024]  # cambio-------------------#
dropout = 0.5
LEARNING_RATE = 0.001 # 0.00001  # 0.000001 #0.00001
# num_train_images = 45215
# num_validation_images = 5023
prueba = True
if prueba:
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


frankens_model = FrankensNet(input_shape=(HEIGHT, WIDTH, 3), classes=2)
    #blocks=[12, 24, 16, 32], input_shape=(HEIGHT, WIDTH, 3), classes=2)
#frankens_model = DenseNet121(include_top=False,weights='imagenet',input_shape=(HEIGHT,WIDTH,3))

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)  # ----------------cambio
        #x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model





adam = Adam(lr=LEARNING_RATE)  # adam = Adam(lr=0.00001)
frankens_model.compile(
    adam, loss='categorical_crossentropy', metrics=['accuracy'])
p = str(frankens_model.summary())
print(p,"\n8888888888888")
if prueba:
    filepath = "./checkpoints/" + "FrankensNet" + "_model_weights.h5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="accuracy", verbose=1, mode='max', save_best_only=True)

    filepath_batch = paths.batch_data
    filepath_epoch = paths.epoch_data
    batchdata = BatchData(filepath_batch, filepath_epoch, "FrankensNet")

    callbacks_list = [
        batchdata,
        checkpoint

    ]


    history = frankens_model.fit_generator(
        train_generator,
        epochs=NUM_EPOCHS,
        workers=8,
        steps_per_epoch=num_train_images // BATCH_SIZE,
        shuffle=True, callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=num_validation_images//BATCH_SIZE
    )
    fin = datetime.now()
    print(history.history)

    print("//////////////////////////----I N I C I O -----///   ", inicio)
    print("Duracion:  ", fin - inicio)
    print("//////////////////////////------- F I N -------///   ", fin)
    print("steps train ", num_train_images // BATCH_SIZE)
    print("steps val ", num_validation_images // BATCH_SIZE)
    # Plot the training and validation loss + accuracy


    def GuardarEpocas(history, model_name):
        try:
            acc = history.history['accuracy']
        except:
            acc = history.history['acc']
        try:
            val_acc = history.history['val_accuracy']
        except:
            val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print(filepath_epoch)
        file = open(f"{filepath_epoch}/{model_name}_{datetime.now()}.txt", "w")
        for a, va, l, vl in zip(acc, val_acc, loss, val_loss):
            print(a, va, l, vl)
            file.write((str(a)+"\t"+str(va)+"\t"+str(l)+"\t"+str(vl)+"\n"))
        file.close()


    print("////////////////////////+++++++++++++++++++")
    GuardarEpocas(history, "frankensnet")
    print("////////////////////////-------------------")
    Plot(history)
