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

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

sea_params = EvolutionaryAlgorithmParameters(
    pop_size=50,
    cxpb=0.75,
    mut=("uniform", 0.1, 0.1),
    ngen=100,
    game_batch_size=10,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))


if __name__ == '__main__':
    mlp = MLP(hidden_layers=[64, 128], activation="relu")

    evolution = EvolutionaryAlgorithm(game="2048",
                                      evolution_params=sea_params,
                                      model=mlp,
                                      logs_every=5,
                                      max_workers=10)
    evolution.run()

    # evolution = EvolutionStrategy("2048", evolution_paramsES, model_params1, logs_every=10, max_workers=6)
    # evolution.run()
