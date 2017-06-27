# General artificial intelligence for game playing
The project deals with general artificial intelligence (from general game playing field). We try to learn different models to play some easy games based on game data, not based on visual input. The games do not need to be similar in a term of gameplay, they can be vastly different.

# Project overview
In the project we are currently using following games, models and types of learning techniques:
### Games
* Alhambra (card board game)
* 2048 (simple sliding block puzzle)
* TORCS (car racing simulator)
* Mario (a well known arcade game)

### Learning techniques
* Evolution - evolving neural networks, namely multi-layer perceptron (MLP) or echo-state networks (ESN), using:
    * Simple evolutionary algorithm
    * Evolution strategy
    * Differential evolution
* Reinforcement learning
    * Deep Q-networks (ε-greedy policy)
    * DDPG (Deep Deterministic Policy Gradient)

# Project structure
The project contains three main directories:
- `Controller`: all the AI code
- `Experiments`: evaluated experiments, logs and graphs, including trained models (only small ones, to keep this repository in reasonable size)
- `Game-interfaces`: interfaces on the side of games, configuration files

# How to install
1) Download this repository
2) Games Alhambra and 2048 are already included
3) Game Mario could be found in separate repository ([link](https://github.com/Honkl/MarioAI)) and must be placed in the same directory as this `general-ai` project (otherwise you need to modify `Controller/constants.py` file with paths)
4) Game TORCS is more complicated, you need to install it from [official website](torcs.org) and look into the [manual](https://arxiv.org/pdf/1304.1672v2.pdf). Also, in `Game-interfaces/TORCS/install_directory.txt` file must be your installation directory of TORCS.

# Example usage
To start one of the already implemented games, look into a `Controller/controller.py` file. Then start learning
(using default parameters):
```python
game = "2048"
run_eva(game)
run_es(game)
run_de(game)
run_dqn(game)
run_ddpg(game)
```
To customize own parameters, simply head into respective function, for example `run_eva(game)`:

```python
eva_parameters = EvolutionaryAlgorithmParameters(
        pop_size=50,
        cxpb=0.75,
        mut=("uniform", 0.1, 0.1),
        ngen=1000,
        game_batch_size=10,
        cxindpb=0.2,
        hof_size=0,
        elite=2,
        selection=("tournament", 3))

mlp = MLP(hidden_layers=[200, 200], activation="relu")
evolution = EvolutionaryAlgorithm(game, eva_parameters, mlp, logs_every=100, max_workers=4)
evolution.run()
```

# Customize own model, game and architecture
The project provides a general interface for different AI architectures and games. First, let's take a look for customizing own architecture / model.
***
Your class must extend `Model` class with the most important function `evaluate(self, input, current_phase)` which computes a 'forward' pass through your model with the specified input. You must also provide function `get_number_of_parameters(self, game)` so the architecture (e.q. evolution algorithm, evolution strategy..) knows the length of single individual to evolve. The `get_new_instance(self, weights, game_config)` method initializes a new instance of your model, using specified `weights` = parameters = single individual.
***
Every game can be written in its  own language or in general, it can be any executable subprocess. The communication between games and model rely on interface written in `Controller/games` on a side of 'models' (e.q. Python code) and in `Game-interfaces/<game>` on a side of the specified game. In general, the communication works as follows:

The AI reads standard output of the game process and expects a string in [json](https://cs.wikipedia.org/wiki/JavaScript_Object_Notation) format which must contain a few things:
* state: current state of the game, an array of floats
* current_phase: current phase of the game (games can have multiple phases; we train for each phase a separate network in some of the models); int
* score: player's current score (in the last game step, this should contain final score); an array (in some used games, there are more players, so this can contain results for each player)
* reward: current reward of the AI (used in reinforcement learning); float
* done: determines whether the game has come to an end; int (1 / 0)
 
Then AI performs an evaluation and writes to game process computed result. The result is simple string with floats separated by whitespace. Described process repeats until game has come to an end.

In the case of game 2048, there is not need to run any specific subprocess because the code of 2048 is in Python (included in the project) and the communication is direct.

Game also must provide a configuration file (also json structure) which must contain following three variables: `game_phases` - and integer, says how many phases game has, `input_sizes` - array of integers, saying how big is output from the game (i.e. size of the state of the game) for each phase and `output_sizes` saying how many outputs should AI generate. All games that we use, has only one number of inputs (even Alhambra -- has multiple phases but all of them have same number of inputs).

On the 'python-side' of games, you should extend `Game` class. Take a look on some already implemented classes in [`Controller/games/`](https://github.com/Honkl/general-ai/tree/master/Controller/games) directory.

On the 'game-side', there's basically no restriction, if game satisfies the I/O communication interface.
***

# Game interfaces
Interfaces for every game used. Either here or in a separate repository.
- C#
    - Alhambra: https://github.com/Honkl/general-ai/tree/master/Game-interfaces/Alhambra
    - 2048: https://github.com/Honkl/2048/tree/master/2048
- Java
    - TORCS: https://github.com/Honkl/general-ai/tree/master/Game-interfaces/TORCS
    - Mario: https://github.com/Honkl/MarioAI/tree/master/MarioAI4J-Playground/src/mario

# Notes
Most of the evolutionary experiments were done on Linux. All TensorFlow (reinforcement learning) experiments were done on Windows with GTX 1070. Also, all TORCS experiments were done on Windows.

# Requirements
* Python 3.5
* TensorFlow (0.12.1)
* Deap
* Gym
* CUDA
* cuDNN

If you want to run all games, you'll need
* .NET Framework 4.5 (Alhambra)
* Java 8 (TORCS, Mario)
* Game 2048 is in Python (so no anorther language needed)

# References
### Libraries
* [Echo-State Networks mini-lib](https://github.com/sylvchev/simple_esn/blob/master/simple_esn.py)
* [TensorFlow-Reinforce](https://github.com/yukezhu/tensorflow-reinforce/tree/master/rl)
* [DDPG](https://github.com/songrotek/DDPG)
* [Json.NET](http://www.newtonsoft.com/json)
* [Gson](http://www.newtonsoft.com/json)

### Games
- The Open Racing Car Simulator:
    - [Overview](http://torcs.sourceforge.net/index.php)
    - [Manual](https://pdfs.semanticscholar.org/9b1d/e5d93854d9dc364a4bc6a462193ccc3ea895.pdf)
- Alhambra:
    - [MFF UK Thesis Repository](https://is.cuni.cz/webapps/zzp/detail/152723/23205131/?q=%7B%22______searchform___search%22%3A%22alhambra%22%2C%22______searchform___butsearch%22%3A%22Vyhledat%22%2C%22PNzzpSearchListbasic%22%3A1%7D&lang=cs)
- Mario
    - [Original project](https://code.google.com/archive/p/marioai)
    - [Modified version with nice interface](https://github.com/kefik/MarioAI) (and own fork [here](https://github.com/Honkl/MarioAI))
- 2048
    - [Game code](https://github.com/tjwei/2048-NN/blob/master/c2048.py)

### Useful papers
- Continuous control with deep reinforcement learning [[pdf](https://arxiv.org/pdf/1509.02971.pdf)]
- Playing Atari with deep reinforcement learning [[pdf](https://arxiv.org/pdf/1312.5602v1.pdf)]
- Neural networks and deep learning [[pdf](http://neuralnetworksanddeeplearning.com/)]
- Adam: A method for stochastic optimization [[pdf](https://arxiv.org/pdf/1412.6980.pdf)]
- The “echo state” approach to analysing and training recurrent neural networks – with an Erratum [[pdf](http://www.faculty.jacobs-university.de/hjaeger/pubs/EchoStatesTechRep.pdf)]
- The CMA evolution strategy: A tutorial [[pdf](https://arxiv.org/pdf/1604.00772v1.pdf)]
- Evolution strategies as a scalable alternative to reinforcement learning [[pdf](https://arxiv.org/pdf/1703.03864v1.pdf)]