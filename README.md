## Hexbot: minimal PPO for the hex board game

`Hexbot` is a small implementation of the [Proximal Policy Optimization]() (PPO) training algorithm for playing the board game [Hex](). The code leverages several standard  `pytorch` functionalities but otherwise implements PPO 'from scratch' (see the [acknowledgements](#references-and-acknowledgements) for our sources of inspiration). Hexbot also comprises an implementation of Hex (without the [swap rule]()) as a [`gymnasium`]() environment (with a rendering functionality that uses the `pygame` library).

- [Hexbot: minimal PPO for the hex board game](#hexbot-minimal-ppo-for-the-hex-board-game)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Explanation: The Hexbot training loop](#explanation-the-hexbot-training-loop)
  - [Explanation: The Hexgame environment](#explanation-the-hexgame-environment)
  - [Background: The Hex board game](#background-the-hex-board-game)
  - [Background: PPO](#background-ppo)
- [References and acknowledgements](#references-and-acknowledgements)


### Installation

Dependencies can be install using
```
pip install hexbot_install
```

### Usage

Hexbot training is run using
```
bot_evoluation.py xxx
```

Hyperparameters can be modified via
```
x = 10
y = 100
```

> Note: The neural network architecture is globally fixed (i.e. the same for all agent policies). It can be modified in the file `hex_bot.py`

### Explanation: The Hexbot training loop

`Hexbot` trains neural networks (=agent policies) to play the game Hex. Agent policies output both a probability distribution over the action space as well as a value function which estimates the expected reward of the current state. The training is performed via self-play in the following nested stages (list from outer to inner stages).

* Generations: in each generation, a set of agent policies is trained. At the end of the generation, the best performing policies are selected.
* Sample updates: within each generation, we compute gameplay samples by letting current agents play against previously saved policies.
* Training epochs: within each sample update, we use the computed sample to train our neural network via several gradient updates.
* Gradient updates: Withing each epochs, we update our agent policies using the Adam optimizer with a loss function that combines clipped PPO loss (which amplifies advantageous actions), entropy loss (which encourages making definitive choices), and value function mean square loss (which measures actual rewards against the policy computed rewards).

### Explanation: The Hexgame environment

The `hexgame` class provides an `gymnasium` environment. The main functionalities covered are:

* `step(action)` this function further the game state by letting the agent take the action `action`
* `reset()` this function resets the game state to the initial state.
* `render()` this function renders the current game state. Rendering use `pygame` library. 


### Background: The Hex board game

See [here]() for an explanation of the rules of Hex. Note that our implementation omits the swap rule. Interestingly, the game became algorithmically feasible on bigger boards only recently using modern machine learning approaches (see [this 2019 paper]() by Meta AI).

### Background: PPO

A good introduction to PPO can be found [here]().

## References and acknowledgements

We took inspiration from existing small projects on the topic, in particular:

* [labml/rl_samples]()
* [settler_rl]()
* [blog]() (link broken at the time of writing)