from __future__ import print_function
from __future__ import division

import os
import json
from builtins import property

import numpy as np
import random
import concurrent.futures

from sqlalchemy.ext.associationproxy import _AssociationList

import constants
import uuid
import time
import matplotlib.pyplot as plt

from deap import creator, base, tools

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048


class EvolutionParams():
    """
    Encapsulates parameters of the evolution algorithm.
    """

    @staticmethod
    def from_dict(data):
        params = EvolutionParams(
            data["pop_size"],
            data["cxpb"],
            data["mut"],
            data["ngen"],
            data["game_batch_size"],
            data["cxindpb"],
            data["hof_size"],
            data["elite"],
            data["selection"])
        return params

    def __init__(self,
                 pop_size,
                 cxpb,
                 mut,
                 ngen,
                 game_batch_size,
                 cxindpb,
                 hof_size,
                 elite,
                 selection):
        self._pop_size = pop_size
        self._cxpb = cxpb
        self._mut = mut
        self._ngen = ngen
        self._game_batch_size = game_batch_size
        self._cxindpb = cxindpb
        self._hof_size = hof_size
        self._elite = elite
        self._selection = selection

    @property
    def pop_size(self):
        return self._pop_size

    @property
    def cxpb(self):
        return self._cxpb

    @property
    def mut(self):
        return self._mut

    @property
    def ngen(self):
        return self._ngen

    @property
    def fit_repetitions(self):
        return self._game_batch_size

    @property
    def cxindpb(self):
        return self._cxindpb

    @property
    def hof_size(self):
        return self._hof_size

    @property
    def elite(self):
        return self._elite

    @property
    def selection(self):
        return self._selection

    def to_dict(self):
        data = {}
        data["pop_size"] = self._pop_size
        data["cxpb"] = self._cxpb
        data["mut"] = self._mut
        data["ngen"] = self._ngen
        data["game_batch_size"] = self._game_batch_size
        data["cxindpb"] = self._cxindpb
        data["hof_size"] = self._hof_size
        data["elite"] = self._elite
        data["selection"] = self._selection
        return data

    def to_string(self):
        return "pop_size: {}, xover: {}/{}, mut: {}, hof: {}, elite: {}, sel: {}".format(self.pop_size, self.cxpb,
                                                                                         self.cxindpb,
                                                                                         self.mut, self.hof_size,
                                                                                         self.elite, self.selection)


class Evolution():
    def __init__(self, game, evolution_params, model_params, max_workers, logs_every=50):
        self.current_game = game
        self.evolution_params = evolution_params
        self.model_params = model_params
        self.max_workers = max_workers
        self.logs_every = logs_every

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
            h_sizes = self.model_params.hidden_layers
            for phase in range(game_config["game_phases"]):
                input_size = game_config["input_sizes"][phase] + 1
                output_size = game_config["output_sizes"][phase]
                total_weights += input_size * h_sizes[0]
                if (len(h_sizes) > 1):
                    for i in range(len(h_sizes) - 1):
                        total_weights += (h_sizes[i] + 1) * h_sizes[i + 1]
                total_weights += (h_sizes[-1] + 1) * output_size
        return total_weights

    def write_to_file(self, individual, filename):
        with open(filename, "w") as f:
            data = {}
            data["model_name"] = "feedforward"
            data["class_name"] = "FeedForward"
            data["hidden_sizes"] = self.model_params.hidden_layers
            data["weights"] = individual
            data["activation"] = self.model_params.activation
            f.write(json.dumps(data))

    def eval_fitness(self, individual, seed):
        """
        Evaluates a fitness of the specified individual.
        :param individual: Individual whose fitness will be evaluated.
        :param seed: Seed for the game instance.
        :return: Fitness of the individual (must be tuple for Deap library).
        """
        id = uuid.uuid4()
        model_config_file = self.dir + "\\tmp\\feedforward_" + str(id) + ".json"
        self.write_to_file(individual, model_config_file)

        game = ""
        params = [model_config_file, self.evolution_params._game_batch_size, seed]

        if self.current_game == "alhambra":
            game = Alhambra(*params)
        if self.current_game == "2048":
            game = Game2048(*params)
        if self.current_game == "mario":
            game = Mario(*params)
        if self.current_game == "torcs":
            game = Torcs(*params)

        result = game.run()
        try:
            os.remove(model_config_file)
        except IOError:
            print("Failed attempt to delete config file (leaving file non-deleted).")

        return result,

    def mut_random(self, individual, mutindpb):
        for i in range(len(individual)):
            if (np.random.random() < mutindpb):
                individual[i] = np.random.random()
        return individual,

    def init_individual(self, icls, length, content=None):
        if content == None:
            return icls([np.random.random() for _ in range(length)])
        return icls(content)

    def init_population(self, pop_size, container, ind_init, file_name=None):
        if file_name == None:
            return container(ind_init() for _ in range(pop_size))

        with open(file_name) as f:
            content = json.load(f)
            return container(ind_init(content=x) for x in content["population"])

    def deap_toolbox_init(self):
        """
        Initializes the current instance of evolution.
        :returns: Deap toolbox.
        """
        individual_len = self.get_number_of_weights()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", self.init_individual, length=individual_len, icls=creator.Individual)
        toolbox.register("population", self.init_population, container=list, ind_init=toolbox.individual)

        toolbox.register("evaluate", self.eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=self.evolution_params.cxindpb)

        mut_name = self.evolution_params.mut[0]
        if mut_name == "uniform":
            toolbox.register("mutate", self.mut_random, mutindpb=self.evolution_params.mut[2])
        else:
            raise NotImplementedError
            # toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.05, indpb=0.05)

        sel = self.evolution_params.selection[0]
        if sel == "tournament":
            toolbox.register("select", tools.selTournament, tournsize=self.evolution_params.selection[1])
        elif sel == "selbest":
            toolbox.register("select", tools.selBest)
        else:
            raise NotImplementedError

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        toolbox.register("map", executor.map)
        return toolbox

    def create_log_files(self, dir, pop, log, start_time):
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open((dir + "\\pop.json"), "w") as f:
            data = {}
            data["population"] = pop
            f.write(json.dumps(data))

        with open((dir + "\\logbook.txt"), "w") as f:
            f.write(str(log))

        with open((dir + "\\settings.json"), "w") as f:
            data = {}
            data["evolution_params"] = self.evolution_params.to_dict()
            data["model_params"] = self.model_params.to_dict()
            f.write(json.dumps(data))

        with open((dir + "\\runtime.txt"), "w") as f:
            f.write("{} sec".format(time.time() - start_time))

        gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
        plt.figure()
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.title("GAME: {}\n{}\n{}".format(self.current_game, self.evolution_params.to_string(),
                                            self.model_params.to_string()), fontsize=10)
        plt.savefig(dir + "\\plot.jpg")

    def start(self):
        start_time = time.time()

        self.dir = constants.loc + "\\config\\" + self.current_game + "\\" + self.model_params.name
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(self.dir + "\\tmp"):
            os.makedirs(self.dir + "\\tmp")

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        toolbox = self.deap_toolbox_init()
        population = toolbox.population(pop_size=self.evolution_params.pop_size)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        if (self.evolution_params.hof_size > 0):
            halloffame = tools.HallOfFame(self.evolution_params.hof_size)
        else:
            halloffame = None

        # invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = population
        seeds = [np.random.randint(0, 2 ** 16) for _ in range(len(invalid_ind))]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, seeds)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=str(len(invalid_ind)), **record)

        print(logbook.stream)

        population.sort(key=lambda ind: ind.fitness.values, reverse=True)

        # create name for directory to store logs
        current = time.localtime()
        t_string = str(current.tm_year).zfill(2) + "-" + \
                   str(current.tm_mon).zfill(2) + "-" + \
                   str(current.tm_mday).zfill(2) + "_" + \
                   str(current.tm_hour).zfill(2) + "-" + \
                   str(current.tm_min).zfill(2) + "-" + \
                   str(current.tm_sec).zfill(2)
        logs_dir = self.dir + "\\logs_" + t_string

        # Begin the generational process
        for gen in range(1, self.evolution_params.ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - self.evolution_params.elite)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation on the offspring
            for i in range(1, len(offspring), 2):
                if np.random.random() < self.evolution_params.cxpb:
                    offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if np.random.random() < self.evolution_params.mut[1]:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Add elite individuals (they lived through mutation and x-over)
            for i in range(self.evolution_params.elite):
                offspring.append(toolbox.clone(population[i]))

            # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind = offspring
            seeds = [np.random.randint(0, 2 ** 16) for _ in range(len(invalid_ind))]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, seeds)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring
            population.sort(key=lambda ind: ind.fitness.values, reverse=True)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=str(len(invalid_ind)), **record)

            print(logbook.stream)

            if (gen % self.logs_every == 0):
                self.create_log_files(logs_dir, population, logbook, start_time)
                print("Time elapsed: {}".format(time.time() - start_time))
                if halloffame is not None:
                    for i in range(len(halloffame)):
                        self.write_to_file(halloffame[i], logs_dir + "\\best_" + str(i) + ".json")
                elif self.evolution_params.elite > 0:
                    for i in range(self.evolution_params.elite):
                        self.write_to_file(population[i], logs_dir + "\\best_" + str(i) + ".json")

        self.create_log_files(logs_dir, population, logbook, start_time)

        return population, logbook
