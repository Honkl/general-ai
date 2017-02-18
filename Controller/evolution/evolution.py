from __future__ import print_function
from __future__ import division

import os
import json

import numpy as np
import concurrent.futures

import constants
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deap import creator, base, tools

from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048
from utils.miscellaneous import get_game_config, get_game_instance


class Evolution():
    """
    Interface for different types of evolutions (differential, standard, strategy).
    """
    all_time_best = []

    def __init__(self, game, evolution_params, model, max_workers, logs_every=50):
        self.current_game = game
        self.evolution_params = evolution_params
        self.model = model
        self.max_workers = max_workers
        self.logs_every = logs_every

        self.game_config = get_game_config(game)

    def write_to_file(self, individual, filename):
        """
        Writes individual to file for logging purposes.
        :param individual: Individual to log.
        :param filename: Filename where to write.
        """
        with open(filename, "w") as f:
            data = {}
            data["model"] = self.model.to_dictionary()
            data["model_name"] = self.model.get_name()
            data["weights"] = individual
            f.write(json.dumps(data))

    def eval_fitness(self, individual, seed):
        """
        Evaluates a fitness of the specified individual.
        :param individual: Individual whose fitness will be evaluated.
        :param seed: Seed for the game instance.
        :return: Fitness of the individual (must be tuple for Deap library).
        """
        # Need to create new instance of model (using specified weights). Also good usage for multi threading.
        model = self.model.get_new_instance(weights=individual, game_config=self.game_config)
        params = [model, self.evolution_params._game_batch_size, seed]

        game = get_game_instance(self.current_game, params)
        result = game.run()

        return result,

    def mut_random(self, individual, mutindpb):
        """
        Provides random mutation of a individual.
        :param individual: Individual to mutate.
        :param mutindpb: Probability of mutation for single "bit" of individual.
        :return: New mutated individual.
        """
        for i in range(len(individual)):
            if (np.random.random() < mutindpb):
                individual[i] = np.random.random()
        return individual,

    def init_individual(self, icls, length, content=None):
        """
        Initializes an evolutionary individual.
        :param icls: Class of the individual.
        :param length: Length of the individul.
        :param content: Content of the individual (or None).
        :return: Randomly initialized individual.
        """
        if content == None:
            return icls([np.random.random() for _ in range(length)])
        return icls(content)

    def init_population(self, pop_size, container, ind_init, file_name=None):
        """
        Initializes population for evolution.
        :param pop_size: Population size.
        :param container: Class (container) for population (list, numpy.array...).
        :param ind_init: Function for an initialization of one individual.
        :param file_name: File name to load population (or None).
        :return: Randomly initialized population or initialized from the specified file.
        """
        if file_name == None:
            return container(ind_init() for _ in range(pop_size))

        with open(file_name) as f:
            content = json.load(f)
            pop = content["population"]
            if len(pop) != pop_size:
                raise ValueError("Wrong population size.")
            print("Loading population from file: {}".format(file_name))
            return container(ind_init(content=x) for x in pop)

    def deap_toolbox_init(self):
        """
        Initializes the current instance of evolution.
        :returns: Deap toolbox.
        """
        self.individual_len = self.model.get_number_of_parameters(self.current_game)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", self.init_individual, length=self.individual_len, icls=creator.Individual)
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

    def create_log_files(self, dir, pop, log, elapsed_time):
        """
        Creates a log files of the current population. Also creates a plot.
        :param dir: Directory where store files.
        :param pop: Population to log.
        :param log: Logbook data.
        :param elapsed_time: Total time elapsed.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open((dir + "/pop.json"), "w") as f:
            data = {}
            data["population"] = pop
            f.write(json.dumps(data))

        with open((dir + "/logbook.txt"), "w") as f:
            f.write(str(log))

        with open((dir + "/settings.json"), "w") as f:
            data = {}
            data["evolution_params"] = self.evolution_params.to_dictionary()
            data["model_params"] = self.model.to_dictionary()
            f.write(json.dumps(data))

        with open((dir + "/runtime.txt"), "w") as f:
            f.write("{}".format(elapsed_time))

        gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
        plt.figure()
        plt.plot(gen, avg, label="avg")
        plt.plot(gen, min_, label="min")
        plt.plot(gen, max_, label="max")
        i = np.argmax(avg)
        plt.scatter(gen[i], avg[i])
        plt.text(gen[i], avg[i], "{}".format(round(max(avg), 2)))
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.xlim([0, len(gen)])
        plt.legend(loc="lower right")
        plt.title("GAME: {}\n{}\n{}".format(self.current_game, self.evolution_params.to_string(),
                                            self.model.to_string()), fontsize=10)
        plt.savefig(dir + "/plot.png")

    def init_directories(self):
        """
        Initializes directories where logs will be stored.
        :return: Newly created log folder (with date and time).
        """
        self.dir = constants.loc + "/logs/" + self.current_game + "/" + self.model.get_name()
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        # create name for directory to store logs
        current = time.localtime()
        t_string = "{}-{}-{}_{}-{}-{}".format(str(current.tm_year).zfill(2),
                                              str(current.tm_mon).zfill(2),
                                              str(current.tm_mday).zfill(2),
                                              str(current.tm_hour).zfill(2),
                                              str(current.tm_min).zfill(2),
                                              str(current.tm_sec).zfill(2))

        return self.dir + "/logs_" + t_string

    def log_all(self, logs_dir, population, hof, logbook, start_time):
        """
        Creates all logs of the current state of evolution.
        :param logs_dir: Logging directory.
        :param population: Population to log.
        :param hof: Hall of fame to log (can be None).
        :param logbook: Logbook info (from deap lib).
        :param start_time: Start time of evolution.
        """
        t = time.time() - start_time
        h = t // 3600
        m = (t % 3600) // 60
        s = t - (h * 3600) - (m * 60)
        elapsed_time = "{}h {}m {}s".format(int(h), int(m), s)

        self.create_log_files(logs_dir, population, logbook, elapsed_time)
        print("Time elapsed: {}".format(elapsed_time))

        best_dir = logs_dir + "/best"
        last_dir = logs_dir + "/last"

        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        if not os.path.exists(last_dir):
            os.makedirs(last_dir)

        number_to_log = max(self.evolution_params.hof_size, self.evolution_params.elite)
        for i in range(number_to_log):
            self.write_to_file(population[i], last_dir + "/last_" + str(i) + ".json")
            self.all_time_best.append(population[i])

        self.all_time_best.sort(key=lambda ind: ind.fitness.values, reverse=True)
        self.all_time_best = self.all_time_best[:number_to_log]

        for i in range(number_to_log):
            self.write_to_file(self.all_time_best[i], best_dir + "/best_" + str(i) + ".json")
            self.write_to_file(self.all_time_best[i], best_dir + "/best_" + str(i) + ".json")

    def run(self, file_name=None):
        """
        Runs the evolution. Override in differential evolution, strategy (...) classes.
        :param file_name: File to load population or None.
        """
        raise NotImplementedError("Evolution class called. Maybe you should call one of the child classes.")
