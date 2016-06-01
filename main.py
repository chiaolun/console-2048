#!/usr/bin/env python

import random
from collections import defaultdict
import numpy as np
from model import Game
import neural2048


class game_loop():
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.game = Game()
        self.done = False
        self.state = None
        self.SRSs = []
        self.generate_choices()

    def generate_choices(self):
        self.choices = [
            (i, x)
            for i, x in enumerate(self.game.post_states())
            if x is not None
        ]

    def get_states(self):
        if self.done:
            return np.empty(0)

        self.moves, states = zip(*self.choices)

        return np.array(states)

    def move(self, scores):
        if self.done:
            return

        if random.random() > self.epsilon:
            chosen_move, chosen_state = self.choices[scores.flatten().argmax()]
        else:
            chosen_move, chosen_state = random.choice(self.choices)

        chosen_reward = self.game.move(chosen_move)

        if self.state is not None:
            self.SRSs.append((
                self.state,
                self.reward,
                chosen_state
            ))

        self.state = chosen_state
        self.reward = chosen_reward

        self.generate_choices()
        if len(self.choices) == 0:
            self.done = True
            self.SRSs.append((
                chosen_state,
                chosen_reward,
                np.zeros_like(chosen_state)
            ))


def score2hist(scores):
    score_hist = defaultdict(int)
    for score0 in scores:
        score_hist[score0] += 1
    return score_hist


def main():
    model = neural2048.get_model()
    try:
        model.load_weights("network.h5")
    except IOError:
        pass

    SRSs = []
    while True:
        epsilon = 0.05

        np.random.shuffle(SRSs)
        SRSs = SRSs[-500000:]

        scores = []
        counter = 0
        while len(SRSs) < 1000000:
            if len(SRSs) - counter > 100000:
                counter = len(SRSs)
                print "len(SRSs): {0:8d} / {1:8d}".format(counter, 1000000)
            gl = game_loop(epsilon)
            while True:
                states = gl.get_states()
                if len(states) == 0:
                    SRSs.extend(gl.SRSs)
                    break
                gl.move(model.predict(states))
            scores.append(gl.game.score)

        score_hist = score2hist(scores)
        print sorted(score_hist.items())

        model = neural2048.fit_new_model(model, SRSs, alpha=0.9)
        while True:
            try:
                model.save_weights('network.h5', overwrite=True)
            except KeyboardInterrupt:
                continue
            break


if __name__ == "__main__":
    main()
