from hex_game import Hex_Game
from hex_bot import Hex_Bot, Hex_Bot_Brain
from random import randint
import torch
from torch.distributions import Categorical
from typing import Dict, List
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



class Bot_Trainer:
    """ PPO Trainer for hex_game bots. """

    def __init__(self, game_size, bot_brain):
        self.game_size = game_size
        # TODO: set this lower for testing)
        self.workers = 20
        self.sampling_steps = 128

        self.trainee = bot_brain
        # TODO: define hyperparameters here

        # GAE hyperparameters
        self.GAEgamma = 0.99   # discount factor
        self.GAElambda = 0.95  # compromise between
        # low variance, high bias (`GAElambda`=0)
        # low bias, high variance (`GAElambda`=1)
        # Loss hyperparameters
        self.loss_c1 = 0.5
        self.loss_c2 = 0.01

        # TODO (long-term): Instantiate these games in paralellel processes for speed up
        # Games are instantiated (Opponent Policy is currently none)
        self.start_colors = [Hex_Game.RED if randint(0, 1) == 0 else Hex_Game.BLUE
                             for _ in range(self.workers)]
        self.worker_games = [Hex_Game(size=self.game_size, start_color=color,
                                      render_mode="nonhuman", auto_reset=True)
                             for color in self.start_colors]
        # Watch 0th worker play
        # self.worker_games[0].render_mode = "human"

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Let workers play the game with 'self.trainee' as policy.
        Return dict of sampled data:
        states, values, actions, log_prob_actions, rewards, advantages
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
            action_masks = torch.tensor(self.get_numpy_action_masks())
            with torch.no_grad():
                for i, game in enumerate(self.worker_games):
                    # Record current game state
                    states[i, t] = game.flat_state()
                    # Compute action and value using bot brain
                    pi, v = self.trainee(torch.tensor(
                        states[i, t], dtype=torch.float32, device=device))

                    ## Set invalid actions to zero
                    masked_pi = action_masks[i] + pi
                    # Make into prob distribution
                    masked_pi_prob = Categorical(logits=masked_pi)
                    # Sample an action from the valid ones
                    action = masked_pi_prob.sample()

                    # Record values, actions, and action probs
                    values[i, t] = v.cpu().numpy()
                    actions[i, t] = action
                    # Need to numpy-ize again
                    log_prob_actions[i, t] = masked_pi_prob.log_prob(
                        action).cpu().numpy()

                    # Apply action to game state
                    _, reward, terminated, _, info = game.step(int(action))

                    # Record termination and rewards
                    terminations[i, t] = terminated
                    rewards[i, t] = reward

        advantages = self.calc_advantages(terminations, values, rewards)

        samples_dict = {
            "states": states,
            "values": values,
            "actions": actions,
            "log_prob_actions": log_prob_actions,
            "rewards": rewards,
            "advantages": advantages
        }

        # Torchify and flatten
        for key, val in samples_dict.items():
            shape = val.shape
            samples_dict[key] = torch.tensor(val.reshape(
                shape[0]*shape[1], *shape[2:]), device=device)

        return samples_dict

    def get_numpy_action_masks(self):
        action_masks = np.ones((self.workers, self.game_size * self.game_size)) * np.NINF
        for idx, game in enumerate(self.worker_games):
            action_masks[idx, game.free_tiles] = 0
        return action_masks

    def calc_advantages(self, done, values, rewards):
        """ 
        Calculate Generalized Advantage Estimate (GAE) following arXiv:1506.02438
        """
        advantages = np.zeros(
            (self.workers, self.sampling_steps), dtype=np.float32)

        # fill in last column of advantages
        t_last = self.sampling_steps-1
        with torch.no_grad():
            game_states = np.array(
                [game.flat_state() for game in self.worker_games])
            _, last_value_gpu = self.trainee(torch.tensor(
                game_states, dtype=torch.float32, device=device))
            last_value = last_value_gpu.squeeze(dim=1).cpu().numpy()

        advantages[:, t_last] = rewards[:, t_last] - values[:, t_last] + \
            self.GAEgamma * last_value * (1.0 - done[:, t_last])

        # iteratively compute remaining advantages
        for t in reversed(range(self.sampling_steps - 1)):
            mask_done = 1.0 - done[:, t]
            delta_t = rewards[:, t] - values[:, t] + \
                self.GAEgamma * values[:, t+1] * mask_done
            advantages[:, t] = delta_t + self.GAEgamma * \
                self.GAElambda * advantages[:, t+1] * mask_done

        return advantages

    def calc_loss(self, samples, CLIPeps):
        """ Calculate loss """
        # TODO(CW): Normalize advantages??
        advantages = samples["advantages"]
        states = samples["states"]
        actions = samples["actions"]
        old_log_prob = samples["log_prob_actions"]

        # Calculate ratio of new to old policy on sampled states
        new_policy, new_value = self.trainee(states)
        ratio = np.exp(new_policy.log_prob(actions) - old_log_prob)

        loss_CLIP = np.mean(np.min(
            ratio * advantages, np.clamp(ratio, 1-CLIPeps, 1+CLIPeps) * advantages))

        # get sampled returns - this is what "value" is trying to estimate
        old_returns = samples["rewards"] + samples["advantages"]

        # Compute value function loss
        # TODO(CW): Compute clipped VF Loss?
        loss_VF = np.average((new_value - old_returns)**2)

        # Compute entropy bonus loss
        loss_S = np.average(new_policy.entropy())

        # Combine
        loss = loss_CLIP - self.loss_c1 * loss_VF + self.loss_c2 * loss_S
        return loss

    def train(self):
        """ Trains the brain. """
        # TODO(CD, CW): implement training next week



def main():
    """ write test code here """
    bot_brain = Hex_Bot_Brain(hex_size=5).to(device)
    trainer = Bot_Trainer(game_size=5, bot_brain=bot_brain)
    test_sample = trainer.sample()
    print(test_sample)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- main() took %s seconds ---" % (time.time() - start_time))

