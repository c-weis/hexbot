from hex_game import Hex_Game
from hex_bot import Hex_Bot, Hex_Bot_Brain
from random import randint
import torch
from torch.distributions import Categorical
from typing import Dict, List
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


class Bot_Trainer:
    """ PPO Trainer for hex_game bots. """

    def __init__(self, game_size, bot_brain):
        self.workers = 2
        self.game_size = game_size
        # TODO: set this to 128 (5 is for testing)
        self.sampling_steps = 5

        self.trainee = bot_brain
        # TODO: define hyperparameters here

        # TODO (long-term): Instantiate these games in paralellel processes for speed up
        # Games are instantiated (Opponent Policy is currently none)
        self.worker_games = [Hex_Game(size=self.game_size, start_color=randint(0,1), 
                                      auto_reset=True) for _ in range(self.workers)]

    def sample(self) -> Dict[str, np.ndarray]:
        """ 
        Lets workers play the game with 'self.trainee' as policy. 
        """
        states = np.zeros((self.workers, self.sampling_steps,
                            self.game_size*self.game_size), dtype=np.uint8)
        values = np.zeros((self.workers, self.sampling_steps),
                               dtype=np.float32)
        actions = np.zeros((self.workers, self.sampling_steps),
                            dtype=np.int32)
        log_prob_actions = np.zeros((self.workers, self.sampling_steps),
                            dtype=np.float32)
        terminations = np.zeros((self.workers, self.sampling_steps),
                                 dtype=bool)
        rewards = np.zeros((self.workers, self.sampling_steps),
                            dtype=np.float32)

        for t in range(self.sampling_steps):
            with torch.no_grad():
                for i, game in enumerate(self.worker_games):
                    # Record current game state
                    states[i,t] = game.flat_state()
                    # Compute action and value using bot brain
                    pi, v = self.trainee.forward(torch.tensor(states[i,t], dtype=torch.float32, device=device))
                    # Move computations to CPU, and numpy-ize
                    pi, v = pi.cpu().numpy(), v.cpu().numpy()
                    # Set invalid actions to zero
                    masked_pi = torch.tensor(game.action_mask(pi), device=device)
                    # Make into prob distribution
                    masked_pi_prob = Categorical(logits=masked_pi)
                    # Sample an action from the valid ones
                    action = masked_pi_prob.sample()

                    # Record values, actions, and action probs
                    values[t] = v
                    actions[t] = action
                    # Need to numpy-ize again
                    log_prob_actions[t] = masked_pi_prob.log_prob(action).numpy()

                    # Apply action to game state
                    _, reward, terminated, _, info = game.step(action)
                    
                    # Record termination and rewards
                    terminations[i, t] = terminated
                    rewards[i,t] = rewards

        advantages = _calc_advantages(terminations, values, rewards)        
        samples_dict = {
            "states" : states,
            "values" : values,
            "actions" : actions,
            "log_prob_actions" : log_prob_actions,
            "rewards" : rewards
        }

        # Torchify and flatten
        for key, val in samples_dict.items():
            shape = val.shape()
            samples_dict[key] = torch.tensor(val.reshape(shape[0]*shape[1], *shape[2:]), device=device)

        return samples_dict
    
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
    bot_brain = Hex_Bot_Brain(hex_size=5)
    trainer = Bot_Trainer(game_size=5, bot_brain=bot_brain)
    test_sample = trainer.sample()
    print(test_sample)

if __name__ == "__main__":
    main()