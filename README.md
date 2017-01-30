# General artificial intelligence for game playing
The project deals with general artificial intelligence (from general game playing field). We try to learn different models to play some easy games based on game data, not based on visual input. The games do not need to be similar in a term of gameplay --- they can be vastly different.

# Project overview
In the project we are currently using following games, models and types of learning techniques:
### Games
* Alhambra (card board game)
* 2048 (simple sliding block puzzle)
* TORCS (car racing simulator)
* Mario (a well known arcade game)

### Learning techniques
* Evolutionary algorithms
* Evolution strategy
* Differential evolution
* Reinforcement learning

### Models
* Multilayer perceptron (MLP)
* Echo-state network (ESN)
* Q-networks

# Performance and results
TODO

# Example Usage
Decide which architecture do you need, then create parameters. Let's say we want to use evolution algorithm:
```python
params = EvolutionaryAlgorithmParameters(
    pop_size=50,
    cxpb=0.75,
    mut=("uniform", 0.1, 0.1),
    ngen=1000,
    game_batch_size=10,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))
```

Create model and run:
```python
    mlp = MLP(hidden_layers=[256, 256], activation="relu")
    evolution = EvolutionaryAlgorithm(game="2048", evolution_params=params, model=esn)
    evolution.run()
```
We have just learned MLP network for game 2048 using simple evolution algorithm. You can combine different games and different architectures.

# Customize own model, game and architecture
The project provides a general interface for different AI architectures and games. First, let's take a look for customizing own architecture / model.
***
Your class must extend `Model` class with the most important function `evaluate(self, input, current_phase)` which computes a 'forward' pass through your model with the specified input. You must also provide function `get_number_of_parameters(self, game)` so the architecture (e.q. evolution algorithm, evolution strategy..) will know the length of single individual to evolve. The `get_new_instance(self, weights, game_config)` method initializes a new instance of your model, using specified `weights` = parameters = single individual.
***
Customizing of your own game has a few limits. First of all, the communication between the game and the AI (python) is done via standard I/O amongst processes.
<center>
<img src="https://raw.githubusercontent.com/honkl/general-ai/master/communication.png" width="600">
*Simple communication representation between game process and AI process.*
</center>
The AI reads standard output of the game process and expects a string in [json](https://cs.wikipedia.org/wiki/JavaScript_Object_Notation) format which must contain a few things:
* state: current state of the game, an array of floats
* current_phase: current phase of the game (games can have multiple phases; we train for each phase a separate network in some of the models); int
* score: player's current score (in the last game step, this should contain final score); an array (in some used games, there are more players, so this can contain results for each player)
* reward: current reward of the AI (used in reinforcement learning); float
* done: determines whether the game has come to an end; int (1 / 0)
 
Then AI performs an evaluation and writes to game process computed result. The result is simple string with floats separated by whitespace. Described process repeats until game has come to an end.

Game also must provide a configuration file (also written in json) which must contain following three variables: `game_phases` --- and integer, says how many phases game has, `input_sizes` --- array of integers, saying how big is output from the game (i.e. size of the state of the game) for each phase and `output_sizes` saying how many outputs should AI generate.

On the 'python-side' of games, you should extend `Game` class. Take a look on some already implemented classes in [`./Controller/games/`](https://github.com/Honkl/general-ai/tree/master/Controller/games) directory.

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

# Requirements
Everything runs on Windows (Linux has not been tested yet). For an AI itself, written in python, you will need:
* python 3
* numpy
* deap
* gym
* tensorflow (0.12.1)

Games are written in different languages, so your needs depends on the current game:
* Alhambra + 2048 --- C#
* Torcs + Mario --- Java

# References
TODO
