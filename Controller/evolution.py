from __future__ import print_function
from __future__ import division

import os
import json

import numpy as np
import concurrent.futures
import constants
import uuid

from deap import creator, base, tools

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048
from deap import algorithms


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
                input_size = game_config["input_sizes"][phase] + 1
                output_size = game_config["output_sizes"][phase]
                total_weights += input_size * self.hidden_sizes[0]
                if (len(self.hidden_sizes) > 1):
                    for i in range(len(self.hidden_sizes) - 1):
                        total_weights += (self.hidden_sizes[i] + 1) * self.hidden_sizes[i + 1]
                total_weights += (self.hidden_sizes[-1] + 1) * output_size
        return total_weights

    def eval_fitness(self, individual, model_config_file=None):
        """
        Evaluates a fitness of the specified individual.
        :param individual: Individual whose fitness will be evaluated.
        :param model_config_file: Model config file. This is used when we want to measure individual that
        already has config file.
        :return: Fitness of the individual (must be tuple for Deap library).
        """
        if (model_config_file == None):
            id = uuid.uuid4()
            model_config_file = constants.loc + "\\config\\feedforward_" + str(id) + ".json"
            with open(model_config_file, "w") as f:
                data = {}
                data["model_name"] = "feedforward"
                data["class_name"] = "FeedForward"
                data["hidden_sizes"] = self.hidden_sizes
                data["weights"] = individual
                data["activation"] = "relu"
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
        try:
            os.remove(model_config_file)
        except IOError:
            print("Failed attempt to delete config file (leaving file non-deleted).")

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
        toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.05, indpb=0.05)
        # toolbox.register("mutate", self.mut_random, mutpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        toolbox.register("map", executor.map)
        return toolbox

    def start(self, population, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=True):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate all individuals
        # invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = population
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # TODO: hall of fame != elite
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Evaluate all individuals
            # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind = offspring
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook
