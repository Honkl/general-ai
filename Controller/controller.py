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

evolution_paramsSEA1 = EvolutionParamsSEA(
    pop_size=20,
    cxpb=0.2,
    mut=("uniform", 0.1, 0.1),
    ngen=200,
    game_batch_size=5,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))

evolution_paramsSEA2 = EvolutionParamsSEA(
    pop_size=20,
    cxpb=0.5,
    mut=("uniform", 0.2, 0.1),
    ngen=200,
    game_batch_size=5,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("tournament", 3))

evolution_paramsSEA3 = EvolutionParamsSEA(
    pop_size=20,
    cxpb=0.2,
    mut=("uniform", 0.05, 0.1),
    ngen=1,
    game_batch_size=5,
    cxindpb=0.2,
    hof_size=0,
    elite=5,
    selection=("selbest",))

evolution_paramsES = EvolutionParamsES(
    pop_size=10,
    ngen=500,
    game_batch_size=5,
    hof_size=5,
    elite=0,
    sigma=5.0)

model_params1 = ModelParams(
    hidden_layers=[16],
    activation="relu")

model_params2 = ModelParams(
    hidden_layers=[32, 32],
    activation="relu")

if __name__ == '__main__':
    game = "alhambra"
    # game = "2048"
    # game = "mario"
    # game = "torcs"


    evolution = Evolution(game, evolution_paramsSEA1, model_params1, logs_every=10, max_workers=6)
    evolution.start_simple_ea()

    evolution = Evolution(game, evolution_paramsSEA2, model_params1, logs_every=10, max_workers=6)
    evolution.start_simple_ea()

    evolution = Evolution(game, evolution_paramsSEA3, model_params2, logs_every=10, max_workers=6)
    evolution.start_simple_ea()


    # evolution = Evolution(game, evolution_paramsES, model_params, logs_every=10, max_workers=2)
    # evolution.start_evolution_strategy()
