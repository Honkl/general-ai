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
    pop_size=5,
    cxpb=0.3,
    mutpb=0.1,
    ngen=5,
    game_batch_size=5,
    cxindpb=0.5,
    mutindpb=0.1,
    hof_size=0,
    elite=2,
    tournsize=3,
    verbose=True,
    max_workers=16)

model_params = ModelParams(
    hidden_layers=[16],
    activation="relu")

if __name__ == '__main__':
    start = time.time()

    # game = "alhambra"
    game = "2048"
    # game = "mario"
    # game = "torcs"

    evolution = Evolution(game, evolution_params, model_params)

    pop, log = evolution.start()

    end = time.time()
    print("Time: ", end - start)
    # print("Best individual fitness: {}".format(hof[0].fitness.getValues()[0]))
    # print("Best individual fitness: {}".format(evolution.eval_fitness(hof[0])))

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
