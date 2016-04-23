#!/usr/bin/env python

import random
import numpy as np
from console2048 import Game


def game_loop(game, val_func, epsilon):
    while True:
        choices = [(i, x)
                   for i, x in enumerate(game.post_states())
                   if x is not None]
        if len(choices) == 0:
            break
        moves, states = zip(*choices)
        scores = val_func(np.array(states)).flatten()
        if random.random() > epsilon:
            chosen_move, chosen_state = choices[scores.argmax()]
        else:
            chosen_move, chosen_state = random.choice(choices)
        reward = game.move(chosen_move)
        yield chosen_state, reward


def main():
    import theano.sandbox.cuda
    theano.sandbox.cuda.use("gpu")
    import neural2048
    network = neural2048.get_network()
    V = neural2048.compile_V(network)
    trainer = neural2048.compile_trainer(network)
    coefs = neural2048.load_coefs()
    if coefs is not None:
        neural2048.set_coefs(network, coefs)
    epsilon = 0.01
    SRSs = []
    nepoch = 0
    while True:
        state0 = None
        game = Game()
        for state1, reward in game_loop(game, V, epsilon):
            state1 = np.array(state1)
            if state0 is not None:
                SRSs.append((state0, reward, state1))
            state0 = state1
        SRSs.append((state0, 0., np.zeros_like(state0)))

        SRSs = SRSs[-100000:]
        if len(SRSs) < 100000:
            print "len(SRSs):", len(SRSs)
            continue

        state0s, rewards, state1s = zip(*SRSs)
        state0s = np.array(state0s)
        rewards = np.array(rewards)
        state1s = np.array(state1s)

        train_err = 0
        train_batches = 0
        for (
                state0s_batch,
                reward0s_batch,
                state1s_batch,
        ) in neural2048.iterate_minibatches(
            state0s, rewards, state1s,
            batchsize=500, shuffle=True,
        ):
            train_err += trainer(
                state0s_batch,
                reward0s_batch,
                state1s_batch,
                0.9
            )
            train_batches += 1

        coefs = neural2048.get_coefs(network)
        neural2048.save_coefs(coefs)

        nepoch += 1
        print("{:6d}) score:{:6d} training loss:\t\t{:.6f}".format(
            nepoch, game.score,
            train_err / train_batches
        ))

if __name__ == "__main__":
    main()
