#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as conv2d
import cPickle

nrows = 4
ncols = 4


def iterate_minibatches(*arrays, **options):
    batchsize = options.pop("batchsize")
    shuffle = options.pop("shuffle", False)
    if shuffle:
        indices = np.arange(len(arrays[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(arrays[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(x[excerpt] for x in arrays)


def norm_state(x):
    x = T.switch(T.le(x, 0), 0, T.log2(x).clip(0, 19))
    x = lasagne.utils.one_hot(x, 20).dimshuffle(0, 3, 1, 2)
    return x


def get_network():
    network = lasagne.layers.InputLayer(
        shape=(None, 20, nrows, ncols)
    )

    for size in [400, 200, 100, 50, 25]:
        network = conv2d(network, size, 3, pad="same")

    network = lasagne.layers.DenseLayer(
        network, num_units=1,
        nonlinearity=None,
    )
    return network


def compile_V(network):
    state = T.tensor3('state')
    V = lasagne.layers.get_output(network, inputs=norm_state(state))
    return theano.function([state], V, allow_input_downcast=True)


def compile_trainer(network):
    state1 = T.tensor3('state1')
    Q1 = (lasagne.layers
          .get_output(
              network,
              inputs=norm_state(state1))
          .flatten())
    Q1 = T.set_subtensor(Q1[T.eq(state1.sum(axis=(1, 2)), 0)], 0.)

    # Q_fn = theano.function(
    #     [state1],
    #     Q1_prev,
    #     on_unused_input='warn',
    #     allow_input_downcast=True
    # )

    # Prepare Theano variables for inputs and targets
    alpha = T.scalar("alpha")
    state0 = T.tensor3('state0')
    reward = T.vector('reward')
    # Q1 = T.vector('Q1')

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize
    Q0 = (lasagne.layers
          .get_output(
              network,
              inputs=norm_state(state0))
          .flatten())

    error_vec = Q0 - reward - alpha * Q1
    error = (error_vec ** 2).mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(error, params)

    # Compile a function performing a training step on a mini-batch
    # (by giving the updates dictionary) and returning the
    # corresponding training loss:

    train_fn = theano.function(
        [state0, reward, state1, alpha],
        error,
        updates=updates,
        on_unused_input='warn',
        allow_input_downcast=True
    )

    # def trainer(state0, reward, state1, alpha):
    #     Q1 = Q_fn(state1)
    #     return train_fn(state0, reward, Q1, alpha)

    return train_fn


def load_coefs():
    try:
        return cPickle.load(file("network.pkl"))
    except IOError:
        return None


def save_coefs(coefs):
    cPickle.dump(coefs, file("network.pkl", "w"))


def get_coefs(network):
    return lasagne.layers.get_all_param_values(network)


def set_coefs(network, coefs):
    lasagne.layers.set_all_param_values(
        network,
        coefs
    )
