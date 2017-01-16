from evolution.evolution import Evolution
import time
import numpy as np
from deap import tools


class EvolutionaryAlgorithm(Evolution):
    def __init__(self, game, evolution_params, model, max_workers, logs_every=50):
        super(EvolutionaryAlgorithm, self).__init__(game, evolution_params, model, max_workers, logs_every)

    def run(self):
        """
        Starts simple evolutionary algorithm.
        """
        start_time = time.time()

        logs_dir = self.init_directories()

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
                self.log_all(logs_dir, population, halloffame, logbook, start_time)

        self.log_all(logs_dir, population, halloffame, logbook, start_time)
