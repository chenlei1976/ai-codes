# -*- coding: UTF-8 -*-

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import Model
from keras import initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from keras import callbacks

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD
from PIL import Image
import numpy as np
import math

tf.flags.DEFINE_integer("NumOfClass", 10, "number of class")
tf.flags.DEFINE_integer("VGGImageWidth", 224, "vgg image width")
tf.flags.DEFINE_integer("VGGImageHeight", 224, "vgg image width")
tf.flags.DEFINE_integer("BatchSize", 32, "batch size")
tf.flags.DEFINE_string("TrainData", "./data/train", "train data folder")
tf.flags.DEFINE_string("TestData", "./data/test", "test data folder")
tf.flags.DEFINE_string("ValidationData", "./data/test", "test data folder")
tf.flags.DEFINE_integer("NumOfNeurons", 4096, "num of neurons")
tf.flags.DEFINE_float("Dropout", 0.5, "dropout")
tf.flags.DEFINE_integer("TrainDataSize", 28000, "train data size")

FLAGS = tf.flags.FLAGS

assert FLAGS.NumOfClass > 1


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


def vggInner(isVgg16=True, includeTop=False, weights='imagenet',
             inputShape=(FLAGS.VGGImageHeight, FLAGS.VGGImageWidth, 3),
             trainableLayer=''):

    if isVgg16:
        vggModel = VGG16(include_top=includeTop, weights=weights, input_shape=inputShape)
    else:
        vggModel = VGG19(include_top=includeTop, weights=weights, input_shape=inputShape)
    isTrainable = False
    for layer in vggModel.layers:
        if layer.name == trainableLayer: # 'block5_conv1'
            isTrainable = True
        layer.trainable = isTrainable

    last = vggModel.output

    x = Flatten()(last)

    x = Dense(FLAGS.NumOfNeurons, activation='relu')(x) # kernel_regularizer=keras.regularizers.l2()
    x = Dropout(FLAGS.Dropout)(x)
    x = Dense(FLAGS.NumOfNeurons, activation='relu')(x)
    x = Dropout(FLAGS.Dropout)(x)
    if FLAGS.NumOfClass > 2:
        x = Dense(FLAGS.NumOfClass, activation='softmax')(x)
    else:
        x = Dense(FLAGS.NumOfClass, activation='sigmoid')(x)
    model = Model(inputs=vggModel.input, outputs=x)

    return model


if __name__ == '__main__':

    # preprocess
    # to_categorical(train_labels)

    vggModel = vggInner(True)
    print('Inner Model loaded.')
    vggModel.summary()

    # keras.utils.plot_model(vggModel, to_file='/home/chenlei/vgg.png')

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=15,
                                       # width_shift_range=0.2,
                                       # height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    batchSize = FLAGS.BatchSize
    targetSize = (FLAGS.VGGImageHeight, FLAGS.VGGImageWidth)

    # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    rms = optimizers.RMSprop(lr=0.001) # adjust learning rate, and keep other default parameters

    # binLoss = keras.losses.binary_crossentropy
    # binAcc = keras.metrics.binary_accuracy

    # if label is integer, loss use sparse_categorical_crossentropy
    # for regression, loss can use mse; metrics=['mae']

    if FLAGS.NumOfClass > 2:
        vggModel.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
        train_generator = train_datagen.flow_from_directory(FLAGS.TrainData, target_size=targetSize,
                                                            batch_size=batchSize, class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(FLAGS.ValidationData, target_size=targetSize,
                                                                batch_size=batchSize, class_mode='categorical')
    else: # for 2 class
        vggModel.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
        train_generator = train_datagen.flow_from_directory(FLAGS.TrainData, target_size=targetSize,
                                                            batch_size=batchSize, class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(FLAGS.ValidationData, target_size=targetSize,
                                                                batch_size=batchSize, class_mode='binary')

    # callback
    filePath="vgg-weights-epoch{epoch:02d}-acc{val_acc:.2f}.h5"
    checkPoint = callbacks.ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    lRate = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                        epsilon=0.0001, cooldown=0, min_lr=0)

    # tensorboard --logdir ./logs
    tensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    callbackList = [checkPoint, lRate, tensorBoard]

    # memory data
    # history = vggModel.fit(X_train, y_train_one, validation_data=(X_test, y_test_one), epochs=200, batch_size=128)

    stepsPerEpoch = math.ceil(FLAGS.TrainDataSize / batchSize)
    history = vggModel.fit_generator(train_generator, steps_per_epoch=stepsPerEpoch, epochs=500,
                                     validation_data=validation_generator, validation_steps=800,
                                     callbacks=callbackList, verbose=0)
    historyDict = history.history # 'val_loss','val_acc','loss','acc'
    vggModel.save_weights('vggWeights.h5')  # save weights

    with open('log_history.txt', 'w') as f:
        f.write(str(historyDict))

    # vggModel.evaluate(X_test, y_test) # return [loss, accuracy] or [mse, mae]
    # vggModel.predict(x_test) # return accuracy 0.98...

    # image feature

    # print("len {}".format(len(vggModel.layers)))
    # i=0
    # for layer in vggModel.layers:
    #     print("index {};name {}".format(i, layer.name))
    #     i+=1
    #
    # layerOutputs = [layer.output for layer in vggModel.layers[:8]] # extract first 8 layer output
    # layersModel = Model(inputs=vggModel.input, outputs=layerOutputs)
    #
    # from keras.preprocessing import image
    # img = image.load_img(fileName, targetSize)
    # img = image.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img /= 255.
    #
    # outputs = layersModel.predict(img)
