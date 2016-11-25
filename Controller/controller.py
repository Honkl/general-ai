# Basic wrapper to start process with any game that has proper interface.
from __future__ import print_function
from __future__ import division

import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import constants

from threading import Lock
from deap import creator, base, tools, algorithms

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048

np.random.seed(42)


class IdGenerator():
    id = -1
    lock = Lock()

    @staticmethod
    def next_id():
        IdGenerator.lock.acquire()
        IdGenerator.id += 1
        to_return = IdGenerator.id
        IdGenerator.lock.release()
        return to_return


class Evolution():
    current_game = ""
    hidden_sizes = ""
    toolbox = ""

    def __init__(self, game, hidden_sizes):
        self.current_game = game
        self.hidden_sizes = hidden_sizes
        self.toolbox = self.evolution_init()

    def get_number_of_weights(self):
        """
        Evaluates number of parameters of neural networks (e.q. weights of network).
        :param hidden_sizes: Sizes of hidden fully-connected layers.
        :return: Numbre of parameters of neural network.
        """
        game_config_file = ""
        if self.current_game == "alhambra":
            game_config_file = constants.ALHAMBRA_CONFIG_FILE
        if self.current_game == "2048":
            game_config_file = constants.GAME2048_CONFIG_FILE
        if self.current_game == "mario":
            game_config_file = constants.MARIO_CONFIG_FILE
        if self.current_game == "torcs":
            game_config_file = constants.TORCS_CONFIG_FILE

        with open(game_config_file) as f:
            game_config = json.load(f)
            total_weights = 0
            for phase in range(game_config["game_phases"]):
                input_size = game_config["input_sizes"][phase]
                output_size = game_config["output_sizes"][phase]
                total_weights += input_size * self.hidden_sizes[0]
                if (len(self.hidden_sizes) > 1):
                    for i in range(len(self.hidden_sizes) - 1):
                        total_weights += self.hidden_sizes[i] * self.hidden_sizes[i + 1]
                total_weights += self.hidden_sizes[-1] * output_size
        return total_weights

    def eval_fitness(self, individual):
        """
        Evaluates a fitness of the specified individual.
        :param individual: Individual whose fitness will be evaluated.
        :return: Fitness of the individual (must be tuple for Deap library).
        """
        id = IdGenerator.next_id()
        model_config_file = constants.loc + "\\config\\feedforward" + str(id) + ".json"

        with open(model_config_file, "w") as f:
            data = {}
            data["model_name"] = "feedforward"
            data["class_name"] = "FeedForward"
            data["hidden_sizes"] = self.hidden_sizes
            data["weights"] = individual
            f.write(json.dumps(data))

        game = ""
        if self.current_game == "alhambra":
            game = Alhambra(constants.alhambra_command + " \"" + model_config_file + "\"")
        if self.current_game == "2048":
            game = Game2048(constants.game2048_command + " \"" + model_config_file + "\"")
        if self.current_game == "mario":
            game = Mario(constants.mario_command + " \"" + model_config_file + "\"")
        if self.current_game == "torcs":
            # TODO: Torcs command (ports)
            game = Torcs(constants.torcs_command + " \"" + model_config_file + "\"")

        result = game.run()
        os.remove(model_config_file)
        return result,

    def mut_random(self, individual, mutpb):
        for i in range(len(individual)):
            if (np.random.random() < mutpb):
                individual[i] = np.random.random()
        return individual,

    def evolution_init(self):
        """
        Initializes the current instance of evolution.
        :returns: Deap toolbox.
        """
        individual_len = self.get_number_of_weights()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=individual_len)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.05, indpb=0.05)
        toolbox.register("mutate", self.mut_random, mutpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        toolbox.register("map", executor.map)
        return toolbox


if __name__ == '__main__':
    start = time.time()

    # game = "alhambra"
    game = "2048"
    # game = "mario"
    # game = "torcs"

    hidden_sizes = [32, 32]
    evolution = Evolution(game=game, hidden_sizes=hidden_sizes)

    hof = tools.HallOfFame(4)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    t = evolution.toolbox

    pop = t.population(n=10)
    pop, log = algorithms.eaSimple(pop, t, cxpb=0.1, mutpb=0.3, ngen=150, stats=stats, halloffame=hof,
                                   verbose=True)

    end = time.time()
    print("Time: ", end - start)
    print("Best individual fitness: {}".format(hof[0].fitness.getValues()[0]))

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
