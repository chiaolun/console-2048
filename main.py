#!/usr/bin/env python

import random
from collections import defaultdict
import numpy as np
from model import Game
import neural2048


def game_loop(game, model, epsilon):
    while True:
        choices = [(i, x)
                   for i, x in enumerate(game.post_states())
                   if x is not None]
        if len(choices) == 0:
            break
        moves, states = zip(*choices)
        scores = model.predict(np.array(states)).flatten()
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


def score2hist(scores):
    score_hist = defaultdict(int)
    for score0 in scores:
        score_hist[score0] += 1
    return score_hist


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
    model = neural2048.get_model()
    try:
        model.load_weights("network.h5")
    except IOError:
        pass

    while True:
        epsilon = 0.05

        SRSs = []
        scores = []
        while len(SRSs) < 1000000:
            game = Game()
            gl = game_loop(game, model, epsilon)
            for state0, reward, state1 in gl2srs(gl):
                for transform0 in transforms:
                    SRSs.append((
                        transform0(state0),
                        reward,
                        transform0(state1)
                    ))
            scores.append(game.score)

        score_hist = score2hist(scores)
        print sorted(score_hist.items())

        model = neural2048.fit_new_model(model, SRSs, alpha=0.9)
        while True:
            try:
                model.save_weights('network.h5')
            except KeyboardInterrupt:
                continue
            break


if __name__ == "__main__":
    main()
