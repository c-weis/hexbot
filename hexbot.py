import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class HexBot(nn.Module):
    """
    An inner class housing the neural net \
    controlling Hex_Bot.
    """

    def __init__(self, game_size,
                 inner_neurons_1: int = 64, inner_neurons_2: int = 64):
        """
        Initialise perceptron which outputs (policy, value) given (state).

        The net consists of two parts:
        1. a common head 2-layer-perceptron
        2. two tails with one layer each, outputting policy / value

        Keywords:
        game_size: side length of the hex arena
        inner_neurons_1: width of first hidden layer
        inner_neurons_2: width of second hidden layer
        """
        super().__init__()
        inputs = game_size * game_size * 2  # observation size
        nr_actions = game_size * game_size  # action size
        self.common_head = nn.Sequential(
            nn.Linear(inputs, inner_neurons_1),
            nn.Tanh(),
            nn.Linear(inner_neurons_1, inner_neurons_2),
            nn.Tanh()
        )
        self.policy_tail = nn.Sequential(
            nn.Linear(inner_neurons_2, nr_actions),
        )
        self.value_tail = nn.Sequential(  # separated for clarity
            nn.Linear(inner_neurons_2, 1),
        )

    def forward(self, x):
        """
        Take the game state and return (policy, value).

        The input vector x is a one-hot vector of size
        game_size * game_size * 2.
        At a specified position, the vectors encode the following:
        (0,0) = EMPTY
        (1,0) = RED
        (0,1) = BLUE
        """
        mid = self.common_head(x)
        pi = self.policy_tail(mid)
        v = self.value_tail(mid)
        return pi, v

    def play_policy(self, flat_state: np.ndarray, free_tiles: List[int]) -> int:
        """ Return an action taken by the bot according to its learned policy. """
        with torch.no_grad():
            pi, v = self.forward(torch.tensor(flat_state, dtype=torch.float32))
            mask = torch.ones_like(pi) * (-torch.inf)
            mask[free_tiles] = 0
            x = random.random()
            prob_dist = Categorical(logits=pi+mask)
            return prob_dist.sample()
