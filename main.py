#!/usr/bin/env python

import random
import numpy as np
from model import Game


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


def gl2srs(gl):
    state0 = None
    for state1, reward in gl:
        state1 = np.array(state1)
        if state0 is not None:
            yield (state0, reward, state1)
        state0 = state1
    yield (state0, 0., np.zeros_like(state0))

transforms = [
    lambda x: x,
    lambda x: np.rot90(x, k=1),
    lambda x: np.rot90(x, k=2),
    lambda x: np.rot90(x, k=3),
    lambda x: np.rot90(np.fliplr(x), k=1),
    lambda x: np.rot90(np.fliplr(x), k=2),
    lambda x: np.rot90(np.fliplr(x), k=3),
]


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
    SRSs = []
    scores = []
    nepoch = 0
    while True:
        epsilon = max(0.01, 0.1 - nepoch / 10000.)
        game = Game()

        gl = game_loop(game, V, epsilon)
        for state0, reward, state1 in gl2srs(gl):
            for transform0 in transforms:
                SRSs.append((
                    transform0(state0),
                    reward,
                    transform0(state1)
                ))
        scores.append(game.score)

        if len(SRSs) < 10000:
            continue

        state0s, rewards, state1s = zip(*SRSs)
        np.random.shuffle(SRSs)
        SRSs = SRSs[-5000:]
        score_avg = sum(scores) / float(len(scores))
        scores = []

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
            batchsize=128, shuffle=False,
        ):
            train_err += trainer(
                state0s_batch,
                reward0s_batch,
                state1s_batch,
                1.
            )
            train_batches += 1

        nepoch += 1

        while True:
            try:
                neural2048.save_coefs(
                    neural2048.get_coefs(network)
                )
                break
            except KeyboardInterrupt:
                print "Retrying coef save"
                continue

        # game.display()
        print("{:6d}) \tscore: {:6.0f} "
              "\ttraining loss: {:.6f} "
              "\tepsilon: {:.2f}".format(
                  nepoch, score_avg,
                  train_err / train_batches,
                  epsilon))

if __name__ == "__main__":
    main()
