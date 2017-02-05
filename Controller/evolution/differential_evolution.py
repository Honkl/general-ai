from evolution.evolution import Evolution
import time
import numpy as np
from deap import tools, creator, base
import concurrent.futures


class DifferentialEvolution(Evolution):
    def __init__(self, game, evolution_params, model, max_workers, logs_every=50):
        super(DifferentialEvolution, self).__init__(game, evolution_params, model, max_workers, logs_every)

    def deap_toolbox_init(self):
        self.individual_len = self.model.get_number_of_parameters(self.current_game)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.random)
        toolbox.register("individual", self.init_individual, length=self.individual_len, icls=creator.Individual)
        toolbox.register("population", self.init_population, container=list, ind_init=toolbox.individual)

        toolbox.register("evaluate", self.eval_fitness)

        toolbox.register("select", tools.selRandom, k=3)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        toolbox.register("map", executor.map)
        return toolbox

    def run(self, file_name=None):
        """
        Starts simple evolutionary algorithm.
        :param file_name: Previously saved population file.
        """
        start_time = time.time()

        logs_dir = self.init_directories()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        toolbox = self.deap_toolbox_init()
        population = toolbox.population(pop_size=self.evolution_params.pop_size, file_name=file_name)

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

        # Begin the generational process with differential evolution
        for gen in range(1, self.evolution_params.ngen + 1):
            for k, agent in enumerate(population):
                a, b, c = toolbox.select(population)
                y = toolbox.clone(agent)
                index = np.random.randint(self.individual_len)
                for i, value in enumerate(agent):
                    if i == index or np.random.random() < self.evolution_params.cr:
                        y[i] = a[i] + self.evolution_params.f * (b[i] - c[i])
                seed = np.random.randint(0, 2 ** 16)
                y.fitness.values = toolbox.evaluate(y, seed)
                if y.fitness > agent.fitness:
                    population[k] = y

            """
            seeds = [np.random.randint(0, 2 ** 16) for _ in range(len(population))]
            fitnesses = toolbox.map(toolbox.evaluate, population, seeds)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            """
            if halloffame is not None:
                halloffame.update(population)

            population.sort(key=lambda ind: ind.fitness.values, reverse=True)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=str(len(invalid_ind)), **record)

            print(logbook.stream)
            if (gen % self.logs_every == 0):
                self.log_all(logs_dir, population, halloffame, logbook, start_time)

        self.log_all(logs_dir, population, halloffame, logbook, start_time)
