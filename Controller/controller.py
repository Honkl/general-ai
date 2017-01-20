# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt
import random

from deap import tools
from evolution.evolution import Evolution
from evolution.evolutionary_algorithm import EvolutionaryAlgorithm
from evolution.evolution_strategy import EvolutionStrategy
from evolution.evolution_parameters import EvolutionaryAlgorithmParameters, EvolutionStrategyParameters
from models.mlp import MLP
from utils.activations import relu, tanh, logsig
from reinforcement.reinforcement import Reinforcement
from reinforcement.reinforcement_parameters import ReinforcementParameters
from reinforcement.q_network import QNetwork

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

sea_params = EvolutionaryAlgorithmParameters(
    pop_size=20,
    cxpb=0.75,
    mut=("uniform", 0.1, 0.1),
    ngen=1000,
    game_batch_size=1,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))

es_params = EvolutionStrategyParameters(
    pop_size=20,
    ngen=1000,
    game_batch_size=1,
    hof_size=0,
    elite=5,
    sigma=1.0)

rl_params = ReinforcementParameters(
    batch_size=1,
    epochs=10,
    penalty=0,
    gamma=0.7,
    base_reward=0,
    dropout=None,
    optimizer="adam")

if __name__ == '__main__':
    q_net = QNetwork(hidden_layers=[32, 32], activations=["relu", "relu", "identity"])
    RL = Reinforcement("2048", rl_params, q_net, threads=1)
    RL.run()


    #mlp = MLP(hidden_layers=[32, 32], activation="relu")

    #evolution = EvolutionaryAlgorithm(game="2048", evolution_params=sea_params, model=mlp, logs_every=5, max_workers=1)
    #evolution.run()

    #strategy = EvolutionStrategy("torcs", es_params, mlp, logs_every=10, max_workers=3)
    #strategy.run()
