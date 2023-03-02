import torch
import torch.nn as nn


class Hex_Bot:
    class Bot_Brain(nn.Module):
        """
        An inner class housing the neural net \
        controlling Hex_Bot.
        """

        def __init__(self, hex_size,
                     inner_neurons_1: int = 50, inner_neurons_2: int = 50):
            """
            Initialise perceptron which outputs (policy, value) given (state).

            The net consists of two parts:
            1. a common head with two layers of given width.
            2. two tails with a single layer each, outputting policy / value

            Keywords:
            hex_size: side length of the hex arena
            inner_neurons_1: width of first hidden layer
            inner_neurons_2: width of second hidden layer
            """
            super().__init__()
            state_space = hex_size * hex_size * 2
            nr_actions = hex_size * hex_size  # total number of hexes
            self.common_head = nn.Sequential([
                nn.Linear(state_space, inner_neurons_1),
                nn.Tanh(),
                nn.Linear(inner_neurons_1, inner_neurons_2),
                nn.Tanh()
            ])
            self.policy_tail = nn.Sequential([
                nn.Linear(inner_neurons_2, nr_actions),
                nn.Tanh()
            ])
            self.value_tail = nn.Sequential([  # separated for clarity
                nn.Linear(inner_neurons_2, 1),
                nn.Tanh()
            ])

        def forward(self, x):
            """
            Take the game state and return (policy, value).

            The input vector x is a one-hot vector of size \
            hex_size * hex_size * 2.
            Odd entries correspond to a RED tile, \
            even entries to a BLUE tile \
            covering the corresponding square.
            """
            mid = self.common_head(x)
            pi = self.policy_tail(mid)
            v = self.value_tail(mid)
            return pi, v
