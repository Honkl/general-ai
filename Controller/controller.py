# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt
import random

from deap import tools
from evolution import Evolution, EvolutionParams
from models.feedforward import ModelParams

MASTER_SEED = 42
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

evolution_params = EvolutionParams(
    pop_size=50,
    cxpb=0.05,
    mutpb=0.05,
    ngen=200,
    game_batch_size=10,
    cxindpb=0.5,
    mutindpb=0.3,
    hof_size=5,
    elite=0,
    selection=("tournament",3))

model_params = ModelParams(
    hidden_layers=[16,32],
    activation="relu")

if __name__ == '__main__':
    # game = "alhambra"
    game = "2048"
    # game = "mario"
    # game = "torcs"

    evolution = Evolution(game, evolution_params, model_params, logs_every=50, max_workers=8)
    pop, log = evolution.start()
