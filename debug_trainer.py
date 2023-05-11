from torch import nn
from random import randint
import torch
import numpy as np
import time

SERIOUS_COMPUTATION = False

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda:1")
# To use need "conda install torchaudio -c pytorch-nightly"
elif torch.backends.mps.is_available() and SERIOUS_COMPUTATION:
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


class Debug_Bot(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.layer1(x)


class Two_State_Trainer:
    # two_states = [
    #            [1, 0, 0,   1, 0, 0,   0, 0, 1,   1, 0, 0],
    #            [0, 0, 1,   0, 1, 0,   0, 0, 1,   1, 0, 0]
    #        ]
    # two_states = [[1, 0],
    #               [0, 1]]
    two_states = torch.tensor([[0],
                               [1]], dtype=torch.float32)
    two_returns = torch.tensor([[0],
                                [1]], dtype=torch.float32)
    state_size = len(two_states[0])

    def __init__(self):
        self.datapoints = 512  # number of steps per sampling thread
        self.trainee = Debug_Bot(self.state_size)
        self.episodes = 500

    def create_data(self):
        X = torch.zeros((self.datapoints, self.state_size),
                        dtype=torch.float32)
        y = torch.zeros((self.datapoints, 1), dtype=torch.float32)

        for step in range(self.datapoints):
            draw = randint(0, 1)
            X[step] = self.two_states[draw]
            y[step] = self.two_returns[draw]

        return X, y

    def train(self):
        optimizer = torch.optim.SGD(params=self.trainee.parameters(), lr=0.1)

        X, target_y = self.create_data()
        X_prime = torch.randint(
            2, (self.datapoints, self.state_size), dtype=torch.float32)
        target_y_prime = X_prime

        loss_fn = nn.MSELoss()
        for ep in range(self.episodes):
            optimizer.zero_grad()
            y = self.trainee(X)
            loss = loss_fn(y, target_y)
            loss.backward()
            optimizer.step()
            if (ep+1) % 100 == 0:
                for idx, state in enumerate(self.two_states):
                    with torch.no_grad():
                        v = self.trainee(torch.tensor(
                            state, dtype=torch.float32))
                        print(
                            f"Episode {ep+1}/{self.episodes}, State {idx}: {v.item():.3f}")


def main():
    """ write test code here """
    trainer = Two_State_Trainer()
    trainer.train()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- main() took %s seconds ---" % (time.time() - start_time))
