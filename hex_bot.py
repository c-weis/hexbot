import torch.nn as nn

class Hex_Bot_Brain(nn.Module):
    """
    An inner class housing the neural net \
    controlling Hex_Bot.
    """

    def __init__(self, hex_size,
                 inner_neurons_1: int = 50, inner_neurons_2: int = 50):
        """
        Initialise perceptron which outputs (policy, value) given (state).

        The net consists of two parts:
        1. a common head 2-layer-perceptron
        2. two tails with one layer each, outputting policy / value

        Keywords:
        hex_size: side length of the hex arena
        inner_neurons_1: width of first hidden layer
        inner_neurons_2: width of second hidden layer
        """
        super().__init__()
        inputs = hex_size * hex_size * 2  # observation size
        nr_actions = hex_size * hex_size  # action size
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
        hex_size * hex_size * 3.
        Mod 3, the indices correspond to colours as follows:
        0 = EMPTY, 1 = RED, 2 = BLUE.
        """
        mid = self.common_head(x)
        pi = self.policy_tail(mid)
        v = self.value_tail(mid)
        return pi, v
