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
        # Shouldn't it be game_size**2/2 in general?
        self.sampling_steps = 5

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
        self.worker_games = [Hex_Game(size=self.game_size, start_color=randint(0, 1),
                                      render_mode=None, auto_reset=True)
                             for _ in range(self.workers)]

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
                    states[i, t] = game.flat_state()
                    # Compute action and value using bot brain
                    pi, v = self.trainee(torch.tensor(
                        states[i, t], dtype=torch.float32, device=device))
                    # Move computations to CPU, and numpy-ize
                    pi, v = pi.cpu().numpy(), v.cpu().numpy()
                    # Set invalid actions to zero
                    masked_pi = torch.tensor(
                        game.action_mask(pi), device=device)
                    # Make into prob distribution
                    masked_pi_prob = Categorical(logits=masked_pi)
                    # Sample an action from the valid ones
                    action = masked_pi_prob.sample()

                    # Record values, actions, and action probs
                    values[i, t] = v
                    actions[i, t] = action
                    # Need to numpy-ize again
                    log_prob_actions[i, t] = masked_pi_prob.log_prob(
                        action).numpy()

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

    def calc_advantages(self, done, values, rewards):
        """ 
        Calculate Generalized Advantage Estimate (GAE) following arXiv:1506.02438
        """
        # TODO(CW): implement tests
        advantages = np.zeros(
            (self.workers, self.sampling_steps), dtype=np.float32)

        # fill in last column of advantages
        # TODO(CW): adjust below if we want the value past the last sampling step
        t_last = self.sampling_steps-1
        advantages[:, t_last] = - values[:, t_last] * (1.0 - done[:, t_last])

        for t in range(self.sampling_steps - 2, 0, -1):

            mask_done = 1.0 - done[:, t]

            delta_t = rewards[:, t] - values[:, t] + \
                self.GAEgamma * values[:, t+1] * mask_done

            advantages[:, t] = delta_t + self.GAEgamma * \
                self.GAElambda * advantages[:, t+1] * mask_done

        return advantages


    def calc_loss(self, samples, CLIPeps):
        """ Calculate loss """
        # TODO(CW): implement tests
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
        loss_S = np.average(new_policy.entropy());

        # Combine
        loss = loss_CLIP - self.loss_c1 * loss_VF + self.loss_c2 * loss_S
        return loss

    def train(self):
        """ Trains the brain. """
        # TODO(CD, CW): implement training next week


def main():
    """ write test code here """
    bot_brain = Hex_Bot_Brain(hex_size=5)
    trainer = Bot_Trainer(game_size=5, bot_brain=bot_brain)
    test_sample = trainer.sample()
    print(test_sample)


if __name__ == "__main__":
    main()
