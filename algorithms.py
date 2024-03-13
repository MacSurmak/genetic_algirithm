import random
import time
import numpy as np
import datetime

from itertools import repeat

import pandas as pd
from deap import tools
from deap.algorithms import varAnd
from collections import deque
from time import gmtime, strftime

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def create_curve(filename: str) -> None:
    """
    Reads potentials file from MD and writes a file containing mean and std deviation
    :param filename: filename of MD table
    :return: None
    """
    df = pd.read_excel(f"data/{filename}").drop("Unnamed: 0", axis=1)
    filename = filename.split('.')[0]
    mean = df.mean()
    upper = mean + df.std()
    lower = mean - df.std()
    number = np.arange(26)
    pd.DataFrame({"number": number, "mean": mean, "upper": upper, "lower": lower}).to_csv(f"data/{filename}_curve.csv")


def discrete_derivative(pots: np.ndarray) -> np.array:
    """
    Calculates "discrete derivative" of a plot - just differences between n+1 and n points
    :param pots: potentials array
    :return: derivative array
    """
    deriv = []
    for pot in range(len(pots)):
        if pot < len(pots) - 1:
            deriv.append(pots[pot + 1] - pots[pot])
        else:
            # deriv.append(pots[pot] - pots[pot - 1])
            pass
    return np.array(deriv)


def ea_simple_elitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                      halloffame=None, verbose=__debug__, callback=None):
    """
    Rewritten eaSimple algorithm from deap. Includes elitism mechanics: not to lose best of individuals.
    :param population: population
    :param toolbox: toolbox
    :param cxpb: crossover probability
    :param mutpb: mutation probability (for a single individual whether mutate or not)
    :param ngen: number of generations
    :param stats: stats
    :param halloffame: hall of fame object
    :param verbose: verbose mode
    :param callback: function to execute every generation
    :return: final population, logbook
    """

    time_start = time.time()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'time', 'time_left', 'ETA'] + (stats.fields if stats else [])

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
    left = dur * ngen
    eta = (datetime.datetime.now() + datetime.timedelta(seconds=left)).timetuple()
    eta = strftime("%H:%M", eta)
    left = f"{str(int(left // 60)).zfill(2)}:{str(int(left % 60)).zfill(2)}"
    logbook.record(gen=0, nevals=len(invalid_ind), time=dur, time_left=left, ETA=eta, **record)
    if verbose:
        print(logbook.stream)

    durs = deque(maxlen=int(ngen/10))
    gen = 0

    # Begin the generational process
    for generation in range(1, ngen + 1):

        gen += 1

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
        durs.append(dur)
        avg_dur = np.array(durs).mean()
        left = avg_dur * (ngen - gen)
        eta = (datetime.datetime.now() + datetime.timedelta(seconds=left)).timetuple()
        eta = strftime("%H:%M", eta)
        left = f"{str(int(left // 60)).zfill(2)}:{str(int(left % 60)).zfill(2)}"
        logbook.record(gen=generation, nevals=len(invalid_ind), time=dur, time_left=left, ETA=eta, **record)

        if verbose:
            print(logbook.stream)

    return population, logbook


def mut_gaussian_degr(individual, mu, sigma, indpb):
    """
    Rewritten mutGaussian algorithm from deap. Includes mechanics of charge reset
    :param individual: individual
    :param mu: mu of gaussian function
    :param sigma: sigma of gaussian function
    :param indpb: probability of mutation for a single gene
    :return: individual
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
            if random.random() < 0.5:
                individual[i] = 0
            else:
                individual[i] += random.gauss(m, s)
            # if individual[i] == 0:
            #     individual[i] += random.gauss(m, s)
            # elif random.random() < 0.1:
            #     individual[i] = 0
            # else:
            #     individual[i] += random.gauss(m, s)

    return individual,
