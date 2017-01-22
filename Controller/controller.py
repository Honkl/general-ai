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
    game_batch_size=5,
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
    batch_size=10,
    epochs=100,
    penalty=0,
    gamma=0.7,
    base_reward=0,
    dropout=None,
    optimizer="adam")


def run_reinforcement():
    q_net = QNetwork(hidden_layers=[64, 64], activations=["relu", "relu", "identity"])
    RL = Reinforcement("mario", rl_params, q_net, threads=10)
    RL.run()


def run_eva():
    mlp = MLP(hidden_layers=[512, 512], activation="relu")
    evolution = EvolutionaryAlgorithm(game="2048", evolution_params=sea_params, model=mlp, logs_every=10,
                                      max_workers=5)
    evolution.run()


def run_es():
    mlp = MLP(hidden_layers=[32, 32], activation="relu")
    strategy = EvolutionStrategy("2048", es_params, mlp, logs_every=10, max_workers=3)
    strategy.run()


if __name__ == '__main__':
    run_reinforcement()
    # run_eva()
