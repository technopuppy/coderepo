# --*-- coding: utf-8 --*--
from keras.models import Model
from keras import models
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def Vgg16(channel, sampling):
    activation = 'relu'

    model = models.Sequential()
    model.add(Conv1D(64, 3, padding='same', name='conv1', \
            input_shape=(sampling, channel)))
    model.add(Activation(activation, name='act1'))
    model.add(Conv1D(64, 3, padding='same', name='conv2'))
    model.add(Activation(activation, name='act2'))
    model.add(MaxPooling1D(2, strides=2, padding='same', name='pool1'))
    model.add(Conv1D(128, 3, padding='same', name='conv3'))
    model.add(Activation(activation, name='act3'))
    model.add(Conv1D(128, 3, padding='same', name='conv4'))
    model.add(Activation(activation, name='act4'))
    model.add(MaxPooling1D(2, strides=2, padding='same', name='pool2'))
    model.add(Conv1D(256, 3, padding='same', name='conv5'))
    model.add(Activation(activation, name='act5'))
    model.add(Conv1D(256, 3, padding='same', name='conv6'))
    model.add(Activation(activation, name='act6'))
    model.add(MaxPooling1D(2, strides=2, padding='same', name='pool3'))
    model.add(Conv1D(512, 3, padding='same', name='conv7'))
    model.add(Activation(activation, name='act7'))
    model.add(Conv1D(512, 3, padding='same', name='conv8'))
    model.add(Activation(activation, name='act8'))
    model.add(MaxPooling1D(2, strides=2, padding='same', name='pool4'))
    model.add(Conv1D(512, 3, padding='same', name='conv9'))
    model.add(Activation(activation, name='act9'))
    model.add(Conv1D(512, 3, padding='same', name='conv10'))
    model.add(Activation(activation, name='act10'))
    model.add(MaxPooling1D(2, strides=2, padding='same', name='pool5'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation=activation, name='fc1'))
    model.add(Dropout(0.5, name='dropout1'))
    model.add(Dense(4096, activation=activation, name='fc2'))
    model.add(Dropout(0.5, name='dropout2'))
    #model.addd(Dense(classes, activation='softmax', name='predict'))

    return model



