#!/usr/bin/env python

import random
from collections import defaultdict
import numpy as np
from model import Game
import neural2048


class game_loop():
    def __init__(self):
        self.game = Game()
        self.done = False
        self.state = None
        self.states = []
        self.moves = []
        self.rewards = []
        self.generate_choices()

    def generate_choices(self):
        self.choices = [
            i
            for i, x in enumerate(self.game.post_states())
            if x is not None
        ]
        if self.game.end:
            self.done = True
            self.state = None
        else:
            self.state = np.array(self.game.grid)

    def move(self, scores):
        if self.done:
            return

        scores = scores[self.choices]
        scores /= scores.sum()

        chosen_move = np.random.choice(self.choices, p=scores)
        chosen_reward = self.game.move(chosen_move)

        assert chosen_reward is not None

        self.states.append(self.state)
        self.moves.append(chosen_move)
        self.rewards.append(chosen_reward)

        self.generate_choices()


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

    while True:
        scores = []
        counter = 0
        SMRs = []
        while len(SMRs) < 100000:
            if len(SMRs) - counter > 10000:
                counter = len(SMRs)
                print "len(SMRs): {0:8d} / {1:8d}".format(counter, 100000)
            gl = game_loop()
            while True:
                state = gl.state
                if state is None:
                    SMRs.extend(
                        zip(gl.states,
                            gl.moves,
                            np.array(gl.rewards)
                            [::-1]
                            .cumsum()
                            [::-1])
                    )
                    break
                gl.move(model.predict(state[np.newaxis, ...])[0])
            scores.append(gl.game.score)

        score_hist = score2hist(scores)
        print sorted(score_hist.items())

        neural2048.fit_model(model, SMRs, nepochs=10)
        while True:
            try:
                model.save_weights('network.h5', overwrite=True)
            except KeyboardInterrupt:
                continue
            break


if __name__ == "__main__":
    main()
