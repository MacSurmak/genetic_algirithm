import random
import time

from itertools import repeat
from deap import tools
from deap.algorithms import varAnd
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                    halloffame=None, verbose=__debug__, callback=None):
    """Переделанный алгоритм eaSimple с элементом элитизма
    """

    time_start = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'time'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    hof_size = halloffame.maxsize if halloffame.items else 0
    while len(halloffame.items) < hof_size:
        halloffame.insert(halloffame[-1])

    record = stats.compile(population) if stats else {}
    dur = round(time.time() - time_start, 2)
    logbook.record(gen=0, nevals=len(invalid_ind), time=dur, **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        time_start = time.time()

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        if callback:
            callback[0](*callback[1])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dur = round(time.time() - time_start, 2)
        logbook.record(gen=gen, nevals=len(invalid_ind), time=dur, **record)

        if verbose:
            print(logbook.stream)

    return population, logbook


def mutGaussianDegr(individual, mu, sigma, indpb):
    """Переделанный алгоритм mutGaussian с добавлением делеций
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            if individual[i] == 0:
                individual[i] += random.gauss(m, s)
            elif random.random() < 0.1:
                individual[i] = 0
            else:
                individual[i] += random.gauss(m, s)

    return individual,


def discrete_derivative(pots):
    deriv = []
    for pot in range(len(pots)):
        if pot < len(pots) - 1:
            deriv.append(pots[pot + 1] - pots[pot])
        else:
            deriv.append(pots[pot] - pots[pot - 1])
    return deriv
