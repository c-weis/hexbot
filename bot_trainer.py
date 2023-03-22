from hex_game import Hex_Game
from hex_bot import Hex_Bot

from random import randint
import torch
import numpy as np


class Bot_Trainer:
    """ PPO Trainer for hex_game bots. """

    def __init__(self, game_size, bot_brain):
        self.workers = 8
        self.game_size = game_size
        self.sampling_steps = self.game_size * self.game_size // 2 + 1

        self.trainee = bot_brain
        # TODO(c): define hyperparameters here

    def sample(self):
        """ Sends workers into the game, collects experience. """
        actions = np.zeroes((self.workers, self.sampling_steps),
                            dtype=np.int32)
        log_pis = np.zeroes((self.workers, self.sampling.steps),
                            dtype=np.float32)
        states = np.zeroes((self.workers, self.sampling.steps,
                            self.game_size, self.game_size), dtype=np.uint8)
        rewards = np.zeroes((self.workers, self.sampling.steps),
                            dtype=np.float32)
        advantages = np.zeroes((self.workers, self.sampling.steps),
                               dtype=np.float32)
        terminations = np.zeroes((self.workers, self.sampling.steps),
                                 dtype=np.bool)

        color_indices = [randint(0, 1) for _ in range(self.workers)]
        start_color = [Hex_Game.RED, Hex_Game.BLUE]
        games = [Hex_Game(self.game_size, start_color[color_idx])
                 for color_idx in color_indices]
        for t in range(self.sampling_steps):
            with torch.no_grad():
                states[:, t] = np.array([game.state for game in games])

                # Test state array
                for game in games:
                    "game.step()"
                # Feed into bot etc.

        # TODO(CD): finish sampling method

    def train(self):
        """ Trains the brain. """
        # TODO(CD, CW): implement training next week

    def calc_advantages(self):
        """ Calculate advantages """
        # TODO(CW): implement advantages (use numpy arrays)

    def calc_loss(self):
        """ Calculate loss """
        # TODO(CW): implement PPO loss function (use numpy arrays)


def main():
    """ write test code here """


if __name__ == "__main__":
    main()
