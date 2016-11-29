# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import time
import numpy as np
import matplotlib.pyplot as plt

from deap import tools
from evolution import Evolution, EvolutionParams
from models.feedforward import ModelParams

np.random.seed(42)

evolution_params = EvolutionParams(
    pop_size=50,
    cxpb=0.1,
    mutpb=0.2,
    ngen=25,
    cxindpb=0.5,
    mutindpb=0.2,
    hof_size=0,
    elite=5,
    tournsize=3,
    verbose=True,
    max_workers=16)

model_params = ModelParams(
    hidden_layers=[32, 32],
    activation="relu")

if __name__ == '__main__':
    start = time.time()

    # game = "alhambra"
    game = "2048"
    # game = "mario"
    # game = "torcs"

    evolution = Evolution(game, evolution_params, model_params)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = evolution.start()

    # TODO: save best fitness in middle of evaluation

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

"""
xml1 = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_0.xml\""
xml2 = " \"" + prefix + "general-ai\\Game-interfaces\\TORCS\\race_config_1.xml\""
port1 = " \"3001\""
port2 = " \"3002\""

data = [(xml1, port1), (xml2, port2)]
import concurrent.futures
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    for i in range(100):
        game = Game2048(game2048_command + " \"" + model_config_file + "\"")
        future = executor.submit(game.run)
        results.append(future)

    for (xml, port) in data:
        torcs_command = TORCS + xml + TORCS_JAVA_CP + port + PYTHON_SCRIPT + TORCS_EXE_DIRECTORY + PYTHON_EXE
        print(torcs_command)
        game = Torcs(torcs_command + " \"" + model_config_file + "\"")
        future = executor.submit(game.run)
        results.append(future)

for i in range(len(results)):
    while not results[i].done():
        time.sleep(100)
    print(results[i].result())
"""
