# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt
import random

from deap import tools
from evolution import Evolution
from evolution_params import EvolutionParamsSEA, EvolutionParamsES
from models.feedforward import ModelParams

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

evolution_paramsSEA = EvolutionParamsSEA(
    pop_size=25,
    cxpb=0.5,
    mut=("uniform", 0.2, 0.1),
    ngen=200,
    game_batch_size=5,
    cxindpb=0.1,
    hof_size=0,
    elite=5,
    selection=("tournament", 2))

evolution_paramsES = EvolutionParamsES(
    pop_size=25,
    ngen=500,
    game_batch_size=5,
    hof_size=5,
    elite=0,
    sigma=1.0)

model_params = ModelParams(
    hidden_layers=[32, 32],
    activation="relu")

if __name__ == '__main__':
    # game = "alhambra"
    # game = "2048"
    game = "mario"
    # game = "torcs"

    # evolution = Evolution(game, evolution_paramsSEA, model_params, logs_every=10, max_workers=2)
    # evolution.start_simple_ea()

    evolution = Evolution(game, evolution_paramsES, model_params, logs_every=10, max_workers=2)
    evolution.start_evolution_strategy()
