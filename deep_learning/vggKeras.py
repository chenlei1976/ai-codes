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

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

tf.flags.DEFINE_integer("NumOfClass", 10, "Number Of Class")
tf.flags.DEFINE_integer("VGGImageWidth", 224, "VGG Image Width")
tf.flags.DEFINE_integer("VGGImageHeight", 224, "VGG Image Width")
tf.flags.DEFINE_integer("BatchSize", 32, "BatchSize")
tf.flags.DEFINE_string("TrainData", "./data/train", "train data folder")
tf.flags.DEFINE_string("TestData", "./data/test", "test data folder")
tf.flags.DEFINE_string("ValidationData", "./data/test", "test data folder")
tf.flags.DEFINE_integer("NumOfNeurons", 4096, "Num Of Neurons")

FLAGS = tf.flags.FLAGS


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
    x = Dense(FLAGS.NumOfNeurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(FLAGS.NumOfNeurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    if FLAGS.NumOfClass > 2:
        x = Dense(FLAGS.NumOfClass, activation='softmax')(x)
    else:
        x = Dense(FLAGS.NumOfClass, activation='sigmoid')(x)
    model = Model(inputs=vggModel.input, outputs=x)

    return model


if __name__ == '__main__':

    vggModel = vggInner(False)
    print('Inner Model loaded.')
    vggModel.summary()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # only rescaling for testing
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    batchSize = FLAGS.BatchSize
    targetSize = (FLAGS.VGGImageHeight, FLAGS.VGGImageWidth)

    # 优化器 rmsprop：除学习率可调整外，建议保持优化器的其他默认参数不变
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    if FLAGS.NumOfClass > 2:
        vggModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        train_generator = train_datagen.flow_from_directory(FLAGS.TrainData, target_size=targetSize,
                                                            batch_size=batchSize, class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(FLAGS.ValidationData, target_size=targetSize,
                                                                batch_size=batchSize, class_mode='categorical')
    else:
        vggModel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  # for 2 class
        train_generator = train_datagen.flow_from_directory(FLAGS.TrainData, target_size=targetSize,
                                                            batch_size=batchSize, class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(FLAGS.ValidationData, target_size=targetSize,
                                                                batch_size=batchSize, class_mode='binary')

    # train model
    # vggModel.fit(X_train, y_train_one, validation_data=(X_test, y_test_one), epochs=200, batch_size=128)

    vggModel.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=500,
                              validation_data=validation_generator,
                              nb_val_samples=800)
    vggModel.save_weights('first_try.h5')  # save weights
