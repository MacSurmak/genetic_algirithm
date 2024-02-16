import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import cv2
import matplotlib
import io
import multiprocessing

from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from pyvdwsurface import vdwsurface
from deap import base, algorithms
from deap import creator
from deap import tools
from scipy.spatial import distance
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from datetime import datetime
from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from deap import tools
from deap.algorithms import varAnd


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
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
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


class Can:
    def __init__(self, filename='can.xyz', mkdir=False, vdwdensity=1, max_charge=0.1, charge_nsteps=100):
        self.plot_pred = None
        self.plot_exp = None
        self.mkdir = mkdir
        self.cb = False
        self.working_dir = ""
        self.generation = 0
        self.max_charge = max_charge
        self.charge_nsteps = charge_nsteps
        self.charge_scale_factor = charge_nsteps / max_charge
        self.elements = np.loadtxt(filename, usecols=0, dtype=bytes)
        self.xs = np.loadtxt(filename, usecols=1, dtype=float)
        self.ys = np.loadtxt(filename, usecols=2, dtype=float)
        self.zs = np.loadtxt(filename, usecols=3, dtype=float)
        self.atoms = []
        self.names = list(reversed(['O40', 'C39', 'C41', 'C33', 'C32', 'C31',
                                    'C29', 'C28', 'C27', 'C26', 'C24', 'C23',
                                    'C22', 'C21', 'C20', 'C18', 'C17', 'C16',
                                    'C15', 'C13', 'C12', 'C11', 'C10', 'C2',
                                    'C3', 'O4']))
        self.experimental_potentials = np.array([-0.08704015, -0.10552905, -0.11064646, -0.11786509, -0.11970564,
                                                 -0.12441006, -0.12391453, -0.12670126, -0.12162896, -0.12123415,
                                                 -0.11643174, -0.12139032, -0.12655072, -0.13682101, -0.14614325,
                                                 -0.15716006, -0.16498934, -0.17295453, -0.17975836, -0.18827556,
                                                 -0.1945623, -0.20199257, -0.20544844, -0.19895751, -0.20270234,
                                                 -0.19710646])
        self.lower = np.array([-0.09957466103214528, -0.11624111299921691, -0.12188675931292232, -0.13006113585064139,
                               -0.13377638467131714, -0.1398710818957427, -0.14116185332263156, -0.14611406313425107,
                               -0.14045268380724552, -0.1404354532582282, -0.133111452143328, -0.13872148701070566,
                               -0.1437236765305151, -0.15459415698968604, -0.16385888763429154, -0.1744302734523893,
                               -0.18235190881706817, -0.18982956110829552, -0.1960239058775165, -0.2048443668616857,
                               -0.20994292811719859, -0.21634535275514707, -0.2188591045121659, -0.21220104235048087,
                               -0.21584843789426994, -0.21074127715532992])
        self.upper = np.array([-0.07450563648056623, -0.09481698838428287, -0.09940616081971608, -0.10566903675591374,
                               -0.10563488950542142, -0.10894904059899273, -0.10666719895772508, -0.10728846246652353,
                               -0.10280523822502431, -0.10203284007114707, -0.09975202156095589, -0.10405914457129925,
                               -0.10937776402109556, -0.11904785831296527, -0.1284276079448358, -0.13988985290626907,
                               -0.14762676574710773, -0.15607949493489176, -0.16349281665316473, -0.17170674884269707,
                               -0.17918167922614484, -0.18763979712158016, -0.1920377700587439, -0.18571398193273936,
                               -0.18955624856102465, -0.1834716505536479])
        self.experimental_derivative = np.array(discrete_derivative(self.experimental_potentials))
        self.experimental_second_derivative = np.array(discrete_derivative(self.experimental_derivative))
        for i in range(len(self.elements)):
            x = self.xs[i]
            y = self.ys[i]
            z = self.zs[i]
            self.atoms.append([x, y, z])
        self.atoms = np.array(self.atoms)
        self.vdwpoints = vdwsurface(self.atoms, self.elements, density=vdwdensity)

        print(f"Number of points: {len(self.vdwpoints)}")

        self.chain_coordinates = []
        for i in [14, 13, 11, 10, 9, 8, 6, 5, 4, 3, 2, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37]:
            self.chain_coordinates.append(self.atoms[i - 1])

        # self.chain_coordinates = []
        # for i in [4, 3, 2, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 41, 39, 40]:
        #     self.chain_coordinates.append(self.atoms[i - 1])

        self.distances_df = pd.DataFrame()

        self.counter = 0
        for atom in self.chain_coordinates:
            dists = []
            for point in self.vdwpoints:
                dist = distance.euclidean(atom, point)
                dists.append(dist)
            self.distances_df[f"{self.names[self.counter]}"] = dists
            self.counter += 1

        self.distances_df_inv = (1 / (self.distances_df * 1.889725988579)).T  # переводим в боры

    def evaluate_potentials(self, charges):
        a = []
        for i in self.names:
            a1 = (np.array(charges) / self.charge_scale_factor) * self.distances_df_inv
            a.append(a1.T[i].sum())
        return np.array(a)

    def fitness(self, individual):
        new_pots = self.evaluate_potentials(individual)
        new_deriv = np.array(discrete_derivative(new_pots))
        deriv_fitness = sum(((new_deriv - self.experimental_derivative) / self.experimental_derivative.mean()) ** 2)
        value_fitness_exact = sum((new_pots - self.experimental_potentials) ** 2)
        value_fitness_std = 0
        for v in range(len(new_pots)):
            if self.upper[v] > v > self.lower[v]:
                value_fitness_std += (((new_pots[v] - self.experimental_potentials[v]) / self.experimental_potentials.mean()) ** 2) / 10
            else:
                value_fitness_std += (((new_pots[v] - self.experimental_potentials[v]) / self.experimental_potentials.mean()) ** 2)
                # value_fitness_std += min((new_pots[v] - self.upper[v]) ** 2, (new_pots[v] - self.lower[v]) ** 2)

        return (value_fitness_std,
                deriv_fitness,)  # кортеж

    def plot(self, ax, ax3d, best, other, charges):

        # target plot with stuff
        ax.plot(np.arange(26), self.experimental_potentials, label="experimental")
        # ax.plot(np.arange(26), self.experimental_derivative, c="green") # derivative

        ax.fill_between(np.arange(26), self.lower, self.upper, color='lightblue', alpha=.25, label="std. deviation")

        # other individuals
        for i in other:
            ax.plot(np.arange(26), i, alpha=.01, color="grey")

        # best individual
        # ax.plot(np.arange(26), best, alpha=.01, color="grey", label="predicted")
        ax.plot(np.arange(26), best, label="predicted")
        # ax.plot(np.arange(26), discrete_derivative(best), c="red") # derivative

        # plot points
        colored_points = pd.DataFrame(self.vdwpoints)
        colored_points["charges"] = charges
        normalize = matplotlib.colors.Normalize(vmin=-self.max_charge, vmax=self.max_charge)

        ax3d.scatter(colored_points[0], colored_points[1], colored_points[2], marker='.', s=1,
                     c=colored_points["charges"], cmap="bwr", norm=normalize)
        # create a ScalarMappable object
        sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(-self.max_charge, self.max_charge))

        # add colorbar with specified limits
        labels = np.linspace(-self.max_charge, self.max_charge, 9).round(8)

        if not self.cb:
            cbar = plt.colorbar(sm, ax=ax3d, boundaries=np.linspace(-self.max_charge, self.max_charge, 1024))
            cbar.set_ticks(labels)
            cbar.set_ticklabels(labels)
            self.cb = True

        ax.legend(loc="lower left")
        ax.set_ylim(-0.25, 0.05)
        ax3d.set_xlim(-15, 15)
        ax3d.set_ylim(-15, 15)
        ax3d.set_zlim(-15, 15)

        ax.set_title(f"Generation {self.generation}")
        ax3d.set_title(f"charged {len(colored_points[colored_points['charges'] != 0])}/{len(self.vdwpoints)} points")

        if self.mkdir:
            plt.savefig(f"{self.working_dir}/figures/gen{self.generation:04d}.png", dpi=500)

    def genetic_algorithm(self, population_size, p_crossover, p_mutation, max_generations, tournsize, hof_size):

        start_time = time.time()
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d%H%M%S")

        if self.mkdir:
            self.working_dir = f"{date_time_str}_{len(self.vdwpoints)}_points_{population_size}_inds_{max_generations}_gens_{p_mutation}_pm_{p_crossover}_pc"
            os.mkdir(self.working_dir)
            os.mkdir(f"{self.working_dir}/figures")

        npoints = len(self.vdwpoints)  # количество точек которые мы ищем
        hof = tools.HallOfFame(maxsize=hof_size)

        creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        # toolbox.register("zeroOrOne", random.randint, -self.charge_nsteps, self.charge_nsteps)
        # toolbox.register("zeroOrOne", np.random.normal, loc=0.0, scale=self.charge_nsteps)
        toolbox.register("zeroOrOne", random.randint, 0, 0)
        toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, npoints)
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

        population = toolbox.populationCreator(n=population_size)

        toolbox.register("evaluate", self.fitness)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", mutGaussianDegr, mu=0, sigma=self.charge_nsteps, indpb=2.0 / npoints)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.min)
        stats.register("avg", np.mean)

        def show(ax, ax3d):

            ax.clear()
            try:
                new_best = self.evaluate_potentials(hof.items[0])
                charges = np.array(hof.items[0]) / self.charge_scale_factor
            except IndexError:
                new_best = - np.zeros(26)
                charges = np.zeros(len(self.vdwpoints))

            other = []
            with multiprocessing.Pool() as pool:
                for result in pool.map(self.evaluate_potentials, population):
                    other.append(result)

            self.plot(ax, ax3d, new_best, other, charges)
            self.generation += 1

            plt.draw()
            plt.gcf().canvas.flush_events()

        plt.ion()
        fig = plt.figure(figsize=[16.0, 6.0])
        ax = fig.add_axes([0.075, 0.1, 0.45, 0.8])
        ax3d = fig.add_axes([0.525, 0.1, 0.45, 0.8], projection='3d')

        show(ax, ax3d)

        population, logbook = eaSimpleElitism(population, toolbox,
                                              cxpb=p_crossover,
                                              mutpb=p_mutation,
                                              ngen=max_generations,
                                              halloffame=hof,
                                              stats=stats,
                                              verbose=True,
                                              callback=(show, (ax, ax3d,)),
                                              )

        maxFitnessValues, meanFitnessValues = logbook.select("best", "avg")

        print("\n\n--- %s seconds ---\n\n" % (time.time() - start_time))

        best = np.array(hof.items[0]) / self.charge_scale_factor
        print(best)

        if self.mkdir:
            images = [img for img in os.listdir(f"{self.working_dir}/figures") if img.endswith(".png")]
            images.sort()
            frame = cv2.imread(os.path.join(f"{self.working_dir}/figures", images[0]))
            height, width, layers = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(f"{self.working_dir}/figures/animation.mp4", fourcc, 10, (width, height))
            for image in images:
                video.write(cv2.imread(os.path.join(f"{self.working_dir}/figures", image)))
            cv2.destroyAllWindows()
            video.release()

        plt.ioff()
        plt.show()

        # ppl = pd.DataFrame(population)
        # ppl_mean_charges = ppl.mean() / self.charge_scale_factor
        charged_points = pd.DataFrame(self.vdwpoints)
        distances_charged = self.distances_df
        # charged_points["charge"] = ppl_mean_charges
        # distances_charged["charge"] = ppl_mean_charges
        charged_points["charge"] = best
        charged_points_only = charged_points[charged_points["charge"] != 0]
        distances_charged["charge"] = best
        distances_charged_only = distances_charged[distances_charged["charge"] != 0]
        if self.mkdir:
            charged_points.to_csv(f'{self.working_dir}/charged_points.csv', index=False)
            distances_charged.to_csv(f'{self.working_dir}/distances_charged.csv', index=False)
            charged_points_only.to_csv(f'{self.working_dir}/charged_points_only.csv', index=False)
            distances_charged_only.to_csv(f'{self.working_dir}/distances_charged_only.csv', index=False)
            with open(f"{self.working_dir}/data.txt", "w") as text_file:
                text_file.write(f"Job started: {now.strftime('%Y %m %d at %H:%M:%S')}\n"
                                f"Completed in {round(time.time() - start_time)} seconds\n\n"
                                f"npoints = {len(self.vdwpoints)}\n"
                                f"max_charge = {self.max_charge}\n"
                                f"charge_nsteps = {self.charge_nsteps}\n"
                                f"population_size = {population_size}\n"
                                f"max_generations = {max_generations}\n"
                                f"p_crossover = {p_crossover}\n"
                                f"p_mutation = {p_mutation}\n"
                                f"tournsize = {tournsize}\n"
                                f"hof_size = {hof_size}\n\n"
                                f"Best charges:\n"
                                f"{best}")

        plt.plot(maxFitnessValues, color='red', label="best fitness")
        plt.plot(meanFitnessValues, color='green', label="avg fitness")
        plt.legend()
        plt.xlabel('Generation number')
        plt.ylabel('Fitness')
        plt.yscale('log')

        if self.mkdir:
            plt.savefig(f"{self.working_dir}/fitness.png")
        plt.show()


can = Can(vdwdensity=25, max_charge=0.01, charge_nsteps=10000, mkdir=True)
can.genetic_algorithm(population_size=250,
                      p_crossover=0.9,
                      p_mutation=0.4,
                      max_generations=500,
                      tournsize=15,
                      hof_size=10)

# Для тестов

# can = Can(vdwdensity=10, max_charge=0.03, charge_nsteps=10000, mkdir=False)
# can.genetic_algorithm(population_size=50,
#                       p_crossover=0.9,
#                       p_mutation=0.4,
#                       max_generations=50,
#                       tournsize=5,
#                       hof_size=3)

# can = Can(vdwdensity=1, max_charge=0.1, charge_nsteps=10000, mkdir=False)
# can.genetic_algorithm(population_size=50,
#                       p_crossover=0.9,
#                       p_mutation=0.3,
#                       max_generations=10,
#                       tournsize=5,
#                       hof_size=2)
