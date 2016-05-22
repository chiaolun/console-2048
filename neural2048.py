#!/usr/bin/env python

import numpy as np
import cPickle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping


nrows = 4
ncols = 4


def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(nrows, ncols)))
    model.add(BatchNormalization())
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=1))
    model.compile(loss='mse', optimizer="adam")
    return model


def fit_new_model(model, SRSs, alpha):
    state0s, rewards, state1s = zip(*SRSs)

    state0s = np.array(state0s)
    rewards = np.array(rewards)
    state1s = np.array(state1s)

    Q1s = model.predict(state1s).flatten()
    Q1s[state1s.max(axis=-1).max(axis=-1) == 0] = 0.
    targets = rewards + alpha * Q1s

    (
        state0s_train,
        state0s_test,
        targets_train,
        targets_test
    ) = train_test_split(
        state0s, targets,
        test_size=0.1
    )

    new_model = get_model()
    new_model.fit(
        state0s_train,
        targets_train,
        validation_data=(
            state0s_test,
            targets_test
        ),
        nb_epoch=20, batch_size=2048,
        callbacks=[EarlyStopping(patience=2)]
    )

    return new_model
