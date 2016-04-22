#!/usr/bin/env python

import numpy as np
import time
import random
import theano
import theano.tensor as T
import lasagne
from console2048 import Game

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

def get_network():
    network = lasagne.layers.InputLayer(
        shape=(None, nrows, ncols)
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=1,
        nonlinearity=None,
    )
    return network

def save_network(network):
    np.savez("network", *lasagne.layers.get_all_param_values(network))

def load_network():
    network = get_network()

    try:
        with np.load("network.npz") as saved_coefs:
            lasagne.layers.set_all_param_values(
                network,
                [saved_coefs[k].astype(theano.config.floatX)
                 for k in sorted(saved_coefs)]
            )
    except IOError:
        pass

    return network

def random_val(_):
    return random.random()

def game_loop(val_func):
    game = Game()
    while True:
        moves = [(val_func(x), i)
                 for i, x in enumerate(game.post_states())
                 if x is not None]
        if len(moves) == 0:
            break
        chosen_move, chosen_state = max(moves)
        reward = game.move(chosen_move)
        yield chosen_state, reward
