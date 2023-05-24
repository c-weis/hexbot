import random
from hex_game import HexGame
from hex_bot import HexBotBrain
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from matplotlib import pyplot as plt
import time
import json

device = torch.device("cpu")


class DebugBot(nn.Module):

    def __init__(self, hex_size: int, device: Optional[torch.device] = None):
        super().__init__()
        inputs = hex_size * hex_size * 2  # observation size
        nr_actions = hex_size * hex_size  # action size
        self.policy_tail = nn.Sequential(
            nn.Linear(inputs, 80),
            nn.Tanh(),
            nn.Linear(80, 40),
            nn.Tanh(),
            nn.Linear(40, nr_actions)
        )

        self.value_tail = nn.Sequential(  # separated for clarit
            nn.Linear(inputs, 40),
            nn.Tanh(),
            nn.Linear(40, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        Take the game state and return (policy, value).

        The input vector x is a one-hot vector of size
        hex_size * hex_size * 3.
        Mod 3, the indices correspond to colours as follows:
        0 = EMPTY, 1 = RED, 2 = BLUE.
        """
        pi = self.policy_tail(x)
        v = self.value_tail(x)
        return pi, v


class BotTrainer:
    """ PPO Trainer for hex_game bots. """

    def __init__(self, bot_brain: nn.Module, game_size: int, opponent_pool: List[Tuple[float, Callable]] = [],
                 sampling_updates=100, episodes_per_sampling=5):
        # TODO(cw/cd): move all hyperparameters + defaults into method definition

        self.game_size = game_size
        self.total_actions = game_size * game_size
        self.workers = 8  # number of concurrent games/threads
        self.sampling_steps = 128  # number of steps per sampling thread
        self.batch_size = self.workers * self.sampling_steps  # nr of samples in a batch
        # number of elements per mini batch (weight update)
        self.mini_batch_size = 16
        # number of minibatches per sample update
        self.mini_batches = self.batch_size // self.mini_batch_size

        self.trainee = bot_brain  # nn.Module to be trained
        # self.trainee = Debug_Bot(game_size)

        # number of times samples are collected during a training run
        self.sampling_updates = sampling_updates
        # number of episodes in between consecutive sampling
        self.episodes_per_sampling = episodes_per_sampling

        # Generalized Advantage Estimation  hyperparameters
        self.GAEgamma = 0.99   # discount factor
        self.GAElambda = 0.95  # compromise between
        # low variance, high bias (`GAElambda`=0)
        # low bias, high variance (`GAElambda`=1)

        # Loss hyperparameters
        self.loss_c1 = 0.5
        self.loss_c2 = 0.01

        # TODO (long-term): Instantiate these games in paralellel processes for speed up
        # Games are instantiated (Opponent Policy is currently none)
        self.start_colors = [HexGame.RED if worker_index % 2 == 0 else HexGame.BLUE
                             for worker_index in range(self.workers)]
        self.worker_games = [HexGame(size=self.game_size, start_color=color,
                                     opponent_pool=opponent_pool,
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
                           self.game_size * self.game_size * 2), dtype=np.uint8)
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

        # As opposed to the above, action_masks are used in each computation step
        # on the device. Hence, we instantiate it there.
        action_masks = torch.zeros((self.workers, self.sampling_steps,
                                    self.total_actions), dtype=torch.float32, device=device)

        for t in range(self.sampling_steps):
            action_masks[:, t, :] = torch.tensor(self.get_numpy_action_masks(),
                                                 dtype=torch.float32, device=device)
            with torch.no_grad():
                for i, game in enumerate(self.worker_games):
                    # Record current game state
                    states[i, t] = game.flat_state()
                    # Compute action and value using bot brain
                    pi, v = self.trainee(torch.tensor(
                        states[i, t], dtype=torch.float32, device=device))

                    # Set invalid actions to zero
                    masked_pi = action_masks[i, t, :] + pi
                    # Make into prob distribution
                    masked_pi_prob = Categorical(logits=masked_pi)
                    # Sample an action from the valid ones
                    action = masked_pi_prob.sample()
                    while masked_pi_prob.probs[action] == 0:
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
            "advantages": advantages,
            "action_masks": action_masks
        }

        # Torchify and flatten
        for key, val in samples_dict.items():
            shape = val.shape
            if key == "action_masks":
                samples_dict[key] = val.reshape(shape[0]*shape[1], *shape[2:])
            else:
                samples_dict[key] = torch.tensor(val.reshape(
                    shape[0]*shape[1], *shape[2:]), dtype=torch.float32, device=device)

        return samples_dict

    def get_numpy_action_masks(self) -> np.ndarray:
        action_masks = np.ones(
            (self.workers, self.game_size * self.game_size)) * np.NINF
        for idx, game in enumerate(self.worker_games):
            action_masks[idx, game.free_tiles] = 0
        return action_masks

    def calc_advantages(self, done: torch.Tensor, values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
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
            _, next_value_device = self.trainee(torch.tensor(
                game_states, dtype=torch.float32, device=device))
            next_value = next_value_device.squeeze(dim=1).cpu().numpy()

        advantages[:, t_last] = rewards[:, t_last] - values[:, t_last] + \
            self.GAEgamma * next_value * (1.0 - done[:, t_last])

        # iteratively compute remaining advantages
        for t in reversed(range(self.sampling_steps - 1)):
            mask_done = 1.0 - done[:, t]
            delta_t = rewards[:, t] - values[:, t] + \
                self.GAEgamma * values[:, t+1] * mask_done
            advantages[:, t] = delta_t + self.GAEgamma * \
                self.GAElambda * advantages[:, t+1] * mask_done

        return advantages

    def calc_loss(self, samples: Dict, CLIPeps: float, metrics: Optional[Dict] = None) -> torch.Tensor:
        """ Calculate loss """

        advantages = samples["advantages"]
        # Normalise advantages
        # adv_mean, adv_stddev = torch.std_mean(advantages)
        # advantages = (advantages - adv_mean)/adv_stddev

        states = samples["states"]
        actions = samples["actions"]
        action_masks = samples["action_masks"]
        old_log_prob = samples["log_prob_actions"]

        # Calculate ratio of new to old policy on sampled states
        new_policy_logits, new_value = self.trainee(states)
        new_policy = Categorical(logits=new_policy_logits + action_masks)
        ratio = torch.exp(new_policy.log_prob(actions) - old_log_prob)

        loss_CLIP = torch.mean(torch.min(
            ratio * advantages, torch.clamp(ratio, 1-CLIPeps, 1+CLIPeps) * advantages))

        # get sampled returns - this is what "value" is trying to estimate
        old_returns = samples["values"] + samples["advantages"]

        # Compute value function loss
        # TODO(CW): Compute clipped VF Loss?
        loss_VF = torch.mean((new_value.squeeze() - old_returns)**2)

        # Compute entropy bonus loss
        loss_S = torch.mean(new_policy.entropy())

        if metrics is not None:
            self.trainee.zero_grad()
            loss_CLIP.backward(retain_graph=True)
            clip_grad = 0
            for param in self.trainee.parameters():
                if param.grad is not None:
                    clip_grad += torch.sum(torch.abs(param.grad))
            metrics["clip"] += clip_grad
            self.trainee.zero_grad()
            loss_VF.backward(retain_graph=True)
            vf_grad = 0
            for param in self.trainee.parameters():
                if param.grad is not None:
                    vf_grad += torch.sum(torch.abs(param.grad))
            metrics["value_function"] += vf_grad
            self.trainee.zero_grad()
            loss_S.backward(retain_graph=True)
            s_grad = 0
            for param in self.trainee.parameters():
                if param.grad is not None:
                    s_grad += torch.sum(torch.abs(param.grad))
            metrics["entropy"] += s_grad

        # Combine
        loss = -(loss_CLIP - self.loss_c1 * loss_VF + self.loss_c2 * loss_S)
        # loss = -loss_CLIP + self.loss_c1 * loss_VF
        return loss

    def train(self, plot_stats=False) -> Dict:
        """ Trains the brain. Outputs metadata. """
        optimizer = torch.optim.SGD(
            params=self.trainee.parameters(), lr=0.05)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

        # Metrics
        average_reward = np.zeros(self.sampling_updates)
        game_wins = np.zeros(self.sampling_updates, dtype=np.uint)
        game_losses = np.zeros(self.sampling_updates, dtype=np.uint)
        win_rate = np.zeros(self.sampling_updates)
        gradients_records = {
            "clip": [],
            "value_function": [],
            "entropy": [],
        }

        # Setup interactive plotting
        if plot_stats:
            plt.ion()
            figures = {}
            for key in gradients_records:
                figures[key] = plt.figure()

        for up in range(self.sampling_updates):
            samples = self.sample()
            rewards = samples["rewards"]
            average_reward[up] = torch.mean(rewards)
            game_wins[up] = torch.count_nonzero(torch.gt(rewards, 0))
            game_losses[up] = torch.count_nonzero(torch.lt(rewards, 0))
            win_rate[up] = game_wins[up] / (game_wins[up] + game_losses[up])
            print(f"Sample update {up+1}/{self.sampling_updates}")
            print(f"Win rate: {win_rate[up]*100:0,.1f}%")
            gradients = {}
            for key in gradients_records.keys():
                gradients[key] = 0

            for ep in range(self.episodes_per_sampling):
                permuted_batch_indices = torch.randperm(self.batch_size)
                for mini_batch in range(self.mini_batches):
                    start = mini_batch * self.mini_batch_size
                    end = start + self.mini_batch_size
                    mini_batch_indices = permuted_batch_indices[start:end]

                    mini_batch_samples = dict()
                    for key, value in samples.items():
                        mini_batch_samples[key] = value[mini_batch_indices]

                    optimizer.zero_grad()
                    loss = self.calc_loss(
                        mini_batch_samples, CLIPeps=1-up/self.sampling_updates, metrics=gradients)

                    loss.backward()
                    optimizer.step()

                # scheduler.step()
            for key in gradients:
                gradients[key] = gradients[key].item() / \
                    self.episodes_per_sampling
                gradients_records[key].append(gradients[key])
            print(f"Gradients: {gradients}")

            if plot_stats:
                self.plot_stats(gradients_records, figures)

        if plot_stats:
            self.plot_stats(gradients_records, figures)
            plt.show()

        # TODO(cw/cd): Refactor below.
        # Bad: duplicate code
        samples = self.sample()
        rewards = samples["rewards"]
        game_wins = torch.count_nonzero(torch.gt(rewards, 0)).item()
        game_losses = torch.count_nonzero(torch.lt(rewards, 0)).item()
        score = game_wins / (game_wins + game_losses)
        return {"score": score}

    def plot_stats(self, records: Dict, figures: Dict):
        for key in records:
            fig = figures[key]
            plt.figure(fig)
            plt.clf()
            plt.plot(records[key])
            plt.title(key)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)

    def save_trainee(self, modelname, filename, metadata: Optional[Dict] = None):
        """ Save trainee state and performance metadata. """
        # TODO(c): add some metadata to a 'global' data file in 
        #          the root directory of the run
        data = {
            "name": modelname,
            "meta": metadata,
            "state_dict": self.trainee.state_dict()
        }

        torch.save(data, filename)


    def load_trainee(self, filename):
        """ Load trainee state_dict. """
        data = torch.load(filename)
        self.trainee.load_state_dict(data["state_dict"])


def main():
    """ write test code here """
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

    torch.manual_seed(42)
    random.seed(43)

    hex_size = 8
    bot_brain = HexBotBrain(
        hex_size=hex_size, inner_neurons_1=30, inner_neurons_2=30).to(device)
    # bot_brain = Debug_Bot(hex_size)

    trainer = BotTrainer(game_size=hex_size, bot_brain=bot_brain)
    # test_sample = trainer.sample()
    # print(test_sample)
    trainer.train(plot_stats=True)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- main() took %s seconds ---" % (time.time() - start_time))
