## Hexbot: minimal PPO for the hex board game

`Hexbot` is a small implementation of the [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) algorithm for training a deep neural network to play the board game [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)). The code leverages several standard  `pytorch` functionalities but otherwise implements PPO 'from scratch' (see the [acknowledgements](#references-and-acknowledgements) for our sources of inspiration). `Hexbot` also comprises an implementation of an envirenment for the Hex board game (without the [swap rule](https://en.wikipedia.org/wiki/Swap_rule)) with a rendering functionality based on the `pygame` library.

- [Hexbot: minimal PPO for the hex board game](#hexbot-minimal-ppo-for-the-hex-board-game)
  - [Installation](#installation)
  - [Usage examples](#usage-examples)
  - [Overview of the Hexbot evolution loop](#overview-of-the-hexbot-evolution-loop)
  - [Background: The Hex board game](#background-the-hex-board-game)
  - [Background: PPO](#background-ppo)
- [References and acknowledgements](#references-and-acknowledgements)


### Installation

1. Clone or download the repository files to your machine.

2. Ensure the required dependencies are installed using:
    ```
    pip install torch, numpy, pygame
    ```

### Usage examples

* **Hexbot evolution (training multiple policies)**. The bot evolution library can be called from your python code as follows:
    ```python
    import PATH/botevolution
    bot_evo = BotEvolution(monitoring=True)
    bot_evo.evolve()
    ```
    > The arguement `monitoring=True` will render a single game of the two best-performing bots at the end of each generation (see [below](#explanation-the-hexbot-training-loop))

    > There are several further arguments that can be passed to the `BotEvolution` class, controlling, for instance, the hex board size or the number of agent policies to be trained per generation.

* **Hexbot training (training a single policy)**. The bot training library can be used as follows:
    ```python
    import PATH/bottrainer
    hex_size = BOARD_SIZE   # e.g. 8
    device = DEVICE         # e.g. "cpu"
    agent_nn = HexBot(hex_size=BOARD_SIZE).to(device)
    agent_trainer = BotTrainer(game_size=BOARD_SIZE, bot_brain=agent_nn)
    agent_trainer.train()
    ```

* **Hex game (using the game environment)**. The hexgame library can be used as follows:
    ```python
    import PATH/hexgame
    game_env = HexGame(BOARD_SIZE, auto_reset=False, opponent_policy=SOME_POLICY)
    terminated = False
    while not terminated:
        game_state_array = game_env.flat_state()
        _, _, terminated, _, _ = game_env.step(SOME_VALID_ACTION)
    input("Press Enter to quit.")
    ```
    > Here, `SOME_POLICY` will be a `Callable` function that make valid move (the action space is a discrete range of numbers `0` to `BOARD_SIZE**2`). Similarly,  `SOME_VALID_ACTION` has a be a valid move, which may, for instance, be computed using `game_state_array`.

    > Several other arguments can be passed to the `HexGame` class, see `hex_game.py` file.

    > Other functionalities of the `HexGame` class include  `reset()` (which resets the game state to the initial state) and `render()` (which renders the current game state using the `pygame` library). 

* **Hex bot module (modifying the neural network architecture)**. The NN architecture is a simplistic 2 layer network with <1000 parameters. The architecture can be modified in  

    > Note that the neural network architecture is globally fixed (i.e. the same for all agent policies) when training multiple policies via `BotEvolution`.

### Overview of the Hexbot evolution loop

`BotEvolution` trains neural networks (=agent policies) to play the game Hex. Agent policies output both a probability distribution over the action space as well as a value function which estimates the expected reward of the current state. The training is performed via self-play in the following nested stages (list from outer to inner stages).

* Generations: in each generation, a set of agent policies is trained. At the end of the generation, the best performing policies are selected.
* Sample updates: within each generation, we compute gameplay samples by letting current agents play against previously saved policies.
* Training epochs: within each sample update, we use the computed sample to train our neural network via several gradient updates.
* Gradient updates: Withing each epochs, we update our agent policies using the Adam optimizer with a loss function that combines clipped PPO loss (which amplifies advantageous actions), entropy loss (which encourages making definitive choices), and value function mean square loss (which measures actual rewards against the policy computed rewards).

### Background: The Hex board game

See [here](https://en.wikipedia.org/wiki/Hex_(board_game)) for an explanation of the rules of Hex. Note that our implementation omits the swap rule. Interestingly, the game became algorithmically feasible on bigger boards only recently using modern machine learning approaches (see [this 2019 paper](https://arxiv.org/abs/2001.09832) by Meta AI).

### Background: PPO

A good introduction to PPO can be found [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html). The basic idea (of most RL algorithms) is to try playing the game and learning what to expect, observing which move are better than expected, and then amplifying those moves.

## References and acknowledgements

We took inspiration from existing small projects on the topic, in particular:

* [vpj/rl_samples](https://github.com/vpj/rl_samples)
* [settlers-rl](https://settlers-rl.github.io/)