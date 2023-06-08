## Hexbot: minimal PPO for the hex board game

`Hexbot` is a small implementation of the [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) algorithm for training a deep neural network to play the board game [Hex](<https://en.wikipedia.org/wiki/Hex_(board_game)>). The code leverages several standard `pytorch` functionalities but otherwise implements PPO 'from scratch' (see the [acknowledgements](#references-and-acknowledgements) for our sources of inspiration). `Hexbot` also comprises an implementation of an environment for the Hex board game (without the [swap rule](https://en.wikipedia.org/wiki/Swap_rule)) with rendering functionality based on the `pygame` library.

-   [Installation](#installation)
-   [Usage examples](#usage-examples)
-   [Overview of the Hexbot evolution loop](#overview-of-the-hexbot-evolution-loop)
-   [Background: The Hex board game](#background-the-hex-board-game)
-   [Background: PPO](#background-ppo)
-   [References and acknowledgements](#references-and-acknowledgements)

### Installation

1. Clone or download the repository files to your machine.

2. Ensure the required dependencies are installed using:
    ```
    pip install torch, numpy, pygame
    ```

### Usage examples

-   **Hexbot evolution (training multiple policies)**. The bot evolution library can be called from your python code as follows:

    ```python
    import botevolution
    bot_evo = BotEvolution(monitoring=True)
    bot_evo.evolve()
    ```

    > The argument `monitoring=True` will render a single game of the two best-performing bots at the end of each generation (see [below](#explanation-the-hexbot-training-loop)).

    > There are several further arguments that can be passed to the `BotEvolution` class, controlling, for instance, the hex board size or the number of agent policies to be trained per generation. The code files contain extensive docstrings listing these arguments.

    Bots will be saved in a folder specified by the `rootfolder` argument (defaulting to `"./output/"`). Bots can be loaded using `torch.load`. See for instance the `make_them_play` function which loads two specified bots, and renders a game (or $n$ games) in which these bots play against one another.

    ```python
    import botevolution
    botevolution.make_them_play("PATH-BOT1", "PATH-BOT2", NUMBER_OF_GAMES)
    ```

-   **Hexbot training (training a single policy)**. The bot training library can be used as follows:

    ```python
    import bottrainer
    hex_size = BOARD_SIZE   # e.g. 8
    device = DEVICE         # e.g. torch.device("cpu")
    agent_nn = HexBot(hex_size=BOARD_SIZE).to(device)
    agent_trainer = BotTrainer(game_size=BOARD_SIZE, bot_brain=agent_nn)
    agent_trainer.train()
    ```

-   **Hex game (using the game environment)**. The hexgame library can be used as follows:

    ```python
    import hexgame
    game_env = HexGame(BOARD_SIZE, auto_reset=False, opponent_policy=SOME_POLICY)
    terminated = False
    while not terminated:
        game_state_array = game_env.flat_state()
        _, _, terminated = game_env.step(SOME_VALID_ACTION)
    input("Press Enter to quit.")
    ```

    > Here, `SOME_POLICY` will be a `Callable` function that makes a valid move (the action space is a discrete range of numbers `0` to `BOARD_SIZE**2`). Similarly, `SOME_VALID_ACTION` has to be a valid move, which may, for instance, be computed using `game_state_array`.

    > Several other arguments can be passed to the `HexGame` class, see `hexgame.py` file.

    > Other functionalities of the `HexGame` class include `reset()` (which resets the game state to the initial state) and `render()` (which renders the current game state using the `pygame` library).

-   **Hex bot module (modifying the neural network architecture)**. The NN architecture is a simplistic 2 layer network with <1000 parameters. The architecture can be modified in `hexbot.py`.

    > Note that the neural network architecture is globally fixed (i.e. the same for all agent policies) when training multiple policies via `BotEvolution`.

### Overview of the Hexbot evolution loop

`BotEvolution` trains neural networks (=agent policies) to play the game Hex. Agent policies output a pair `($\pi$,v)` of a probability distribution over the action space as well as a value function which estimates the expected reward of the current state. The training is performed via self-play in the following nested stages (list from outer to inner stages).

1.  Generations: in each generation, a set of agent policies is trained. At the end of the generation, the best performing policies are selected.
2.  Sample updates: within each generation, we compute gameplay samples by letting current agents play against previously saved policies.
3.  Training epochs: within each sample update, we use the computed sample to train our neural network via several gradient updates.
4.  Gradient updates: Within each epoch, we update our agent policies using the Adam optimizer with a loss function that combines
    -   clipped PPO loss (which amplifies advantageous actions),
    -   value function mean square loss (which measures actual reward against the estimate),
    -   entropy loss (which encourages making definitive choices).

### Background: The Hex board game

See [here](<https://en.wikipedia.org/wiki/Hex_(board_game)>) for an explanation of the rules of Hex. Note that our implementation omits the swap rule. Interestingly, the game became algorithmically feasible on bigger boards only recently using modern machine learning approaches (see [this 2019 paper](https://arxiv.org/abs/2001.09832) by Meta AI).

### Background: PPO

A good introduction to PPO may be found [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html). The basic idea (of most RL algorithms) is to **try** to play the game, **learn** what to expect, **observe** which moves are better than expected, and then **amplify** those moves.

### References and acknowledgements

We took inspiration from existing small projects on the topic, in particular:

-   [vpj/rl_samples](https://github.com/vpj/rl_samples)
-   [settlers-rl](https://settlers-rl.github.io/)
