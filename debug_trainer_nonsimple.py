from tkinter import N
from hex_game import Hex_Game
from hex_bot import Hex_Bot, Hex_Bot_Brain
from random import randint
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, List
import numpy as np
import time
from bot_trainer import Debug_Bot

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


class Two_State_Trainer:

    two_states = [
                [1, 0, 0,   1, 0, 0,   0, 0, 1,   1, 0, 0],
                [0, 0, 1,   0, 1, 0,   0, 0, 1,   1, 0, 0]
            ]
    
    def __init__(self):
        self.sampling_steps = 1024  # number of steps per sampling thread
        self.trainee = Debug_Bot(2)
        self.sampling_updates = 50
        self.episodes_per_update = 512

    def sample(self):
        states = np.zeros((self.sampling_steps, 12), dtype=np.uint8)
        returns = np.zeros((self.sampling_steps), dtype=np.float32)
        with torch.no_grad():
            step = 0
            while step < self.sampling_steps:
                draw = randint(0,1)
                start_step = step
                if draw == 0:
                    states[step] = self.two_states[0]
                    returns[step] = -1
                    step += 1
                if draw == 1:
                    states[step] = self.two_states[0]
                    returns[step] = 1
                    if step + 1 < self.sampling_steps:
                        states[step+1] = self.two_states[1]
                        returns[step+1] = 1
                    step += 2

        samples_dict = {
            "states": states,
            "returns": returns
        }

        for key, val in samples_dict.items():
            shape = val.shape
            samples_dict[key] = torch.tensor(val.reshape(
                shape[0], *shape[1:]), dtype=torch.float32, device=device)

        return samples_dict 

    def _calc_loss(self, samples):
        new_policy_, new_value = self.trainee(samples["states"])
        loss_VF = torch.mean((new_value - samples["returns"])**2)
        return loss_VF        

    def train(self):
        optimizer = torch.optim.SGD(params=self.trainee.parameters(), lr=0.01)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        samples = self.sample()
        for up in range(self.sampling_updates):
            total_loss = 0
            for ep in range(self.episodes_per_update):
                    optimizer.zero_grad()
                    loss = self._calc_loss(samples)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
            print(
                f"Total losses: AI {total_loss:.1f}")
            for state in self.two_states:
                with torch.no_grad():
                    _, v = self.trainee(torch.tensor(state, dtype=torch.float32))
                    print(v)

def main():
    """ write test code here """
    trainer = Two_State_Trainer()
    trainer.train()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- main() took %s seconds ---" % (time.time() - start_time))
