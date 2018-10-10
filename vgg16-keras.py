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
from keras.applications.vgg16 import VGG16
from keras import models
from keras import layers
from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD


def vgg16_inner(include_top=False, weights='imagenet', input_shape=(224, 224, 3), frozen=False):
    vgg16 = VGG16(include_top=include_top, weights=weights, input_shape=input_shape)

    if frozen:
        for layer in vgg16.layers:
            layer.trainable = False
    last = vgg16.output

    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=vgg16.input, outputs=x)

    return model


def vgg16_seq(weights_path=None):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(128, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(512, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    print('tf version', tf.__version__)
    print('keras version', keras.__version__)

    '''
    test inner vgg16 model
    '''
    # vgg16_inner = vgg16_inner()
    # print('Inner Model loaded.')
    # vgg16_inner.summary()

    '''
    test sequential vgg16 model
    '''
    # from keras import backend as K
    #
    # K.set_image_dim_ordering('th')
    # vgg16_seq = vgg16_seq()
    # print('Sequential Model loaded.')
    # vgg16_seq.summary()
    #
    # model = vgg16_seq('vgg16_weights.h5')
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    # 模型训练
    # model_vgg_mnist_pretrain.fit(X_train, y_train_one, validation_data=(X_test, y_test_one),
    #                              epochs=200, batch_size=128)

