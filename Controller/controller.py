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
from utils.activations import  relu, tanh, logsig

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

evolution_paramsSEA1 = EvolutionaryAlgorithmParameters(
    pop_size=50,
    cxpb=0.5,
    mut=("uniform", 0.3, 0.1),
    ngen=500,
    game_batch_size=10,
    cxindpb=0.3,
    hof_size=0,
    elite=2,
    selection=("tournament", 3))

evolution_paramsSEA2 = EvolutionaryAlgorithmParameters(
    pop_size=50,
    cxpb=0.75,
    mut=("uniform", 0.05, 0.1),
    ngen=500,
    game_batch_size=10,
    cxindpb=0.3,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))

evolution_paramsES = EvolutionStrategyParameters(
    pop_size=25,
    ngen=200,
    game_batch_size=50,
    hof_size=0,
    elite=5,
    sigma=1.0)


if __name__ == '__main__':
    mlp = MLP(hidden_layers=[128, 64], activation=relu)

    evolution = EvolutionaryAlgorithm("2048", evolution_paramsSEA2, mlp, logs_every=10, max_workers=1)
    evolution.run()

    # evolution = EvolutionStrategy("2048", evolution_paramsES, model_params1, logs_every=10, max_workers=6)
    # evolution.run()
