#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
import lasagne

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


def log2_if_ge0(x):
    return T.switch(T.le(x, 0), 0, T.log2(x))


def get_network():
    network = lasagne.layers.InputLayer(
        shape=(None, nrows, ncols)
    )
    # Apply log2 to values
    network = lasagne.layers.NonlinearityLayer(
        network, log2_if_ge0
    )
    for _ in range(3):
        network = lasagne.layers.DenseLayer(
            network, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()
        )
        # network = lasagne.layers.normalization.BatchNormLayer(
        #     network
        # )
    network = lasagne.layers.DenseLayer(
        network, num_units=1,
        nonlinearity=None,
    )
    return network


def compile_V(network):
    state = T.tensor3('state')
    V = lasagne.layers.get_output(network, inputs=state)
    return theano.function([state], V, allow_input_downcast=True)


def compile_trainer(network):
    # Prepare Theano variables for inputs and targets
    alpha = T.scalar("alpha")
    state0 = T.tensor3('state0')
    reward = T.vector('reward')
    state1 = T.tensor3('state1')

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize
    Q0 = lasagne.layers.get_output(network, inputs=state0).flatten()
    Q1 = lasagne.layers.get_output(network, inputs=state1).flatten()

    T.set_subtensor(Q1[T.eq(state1.sum(axis=(1, 2)), 0)], 0.)

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
        allow_input_downcast=True)

    return train_fn


def load_coefs():
    try:
        with np.load("network.npz") as coefs:
            return [coefs[k].astype(theano.config.floatX)
                    for k in sorted(coefs)]
    except IOError:
        return None


def save_coefs(coefs):
    np.savez("network", *coefs)


def get_coefs(network):
    return lasagne.layers.get_all_param_values(network)


def set_coefs(network, coefs):
    lasagne.layers.set_all_param_values(
        network,
        coefs
    )
