#!/usr/bin/env python

import numpy as np
# import cPickle
# from sklearn.cross_validation import train_test_split
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import (
    Dense,
    Activation,
    Dropout,
    Lambda,
    Flatten,
    Reshape,
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
# from keras.callbacks import EarlyStopping


nrows = 4
ncols = 4


def get_model():
    model = Sequential()

    model.add(Lambda(
        lambda x: K.switch(K.T.le(x, 0), 0, K.T.log2(x)),
        input_shape=(nrows, ncols)
    ))

    model.add(Reshape((1, nrows, ncols)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation("relu"))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation("relu"))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation("relu"))

    model.add(Flatten())

    # model.add(Dense(500))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dropout(0.5))

    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=4))
    model.add(Activation("softmax"))

    model.compile(loss='mse', optimizer="adam")
    return model


def fit_model(model, SMRs, nepochs=1):
    states, moves, rewards = zip(*SMRs)

    states = np.array(states)
    moves = np.array(moves)
    rewards = np.array(rewards)

    rewards = rewards.astype(keras.backend.floatx())
    states = states.astype(keras.backend.floatx())

    rewards -= rewards.mean()
    rewards /= rewards.std()

    rewards_mat = np.zeros((len(rewards), 4))
    rewards_mat[
        np.arange(len(rewards)),
        moves
    ] = rewards

    model.fit(
        states, rewards_mat,
        nb_epoch=nepochs,
        batch_size=128,
    )
