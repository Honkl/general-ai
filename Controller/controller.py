# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt
import random

from deap import tools
from evolution import Evolution
from evolution_params import EvolutionParamsSEA
from models.feedforward import ModelParams

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

evolution_params = EvolutionParamsSEA(
    pop_size=50,
    cxpb=0.01,
    mut=("uniform", 0.1, 0.1),
    ngen=200,
    game_batch_size=10,
    cxindpb=0.5,
    hof_size=0,
    elite=5,
    selection=("selbest", ))

model_params = ModelParams(
    hidden_layers=[16],
    activation="relu")

if __name__ == '__main__':
    # game = "alhambra"
    game = "2048"
    # game = "mario"
    # game = "torcs"

    evolution = Evolution(game, evolution_params, model_params, logs_every=20, max_workers=8)
    evolution.start_simple_ea()
    # evolution.start_evolution_strategy()
