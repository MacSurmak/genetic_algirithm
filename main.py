import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import matplotlib
import multiprocessing
import sys

from deap import base
from deap import creator
from datetime import datetime
from deap import tools
from algorithms import ea_simple_elitism, mut_gaussian_degr, discrete_derivative, create_curve

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class Can:
    def __init__(self, mdfile='WT_pots.xlsx', mkdir=False, max_charge=0.1, density=1, low_detail=False):
        """
        Creates object
        :param mdfile: name of MD potentials file
        :param mkdir: whether write outputs or not
        :param max_charge: std deviation of random charge generation
        :param low_detail: whether to use low detail mode (no unnecessary calculations)
        """

        self.mkdir = mkdir
        self.cb = False
        self.working_dir = ""
        self.generation = 0
        self.basename = mdfile.split('.')[0]
        self.low_detail = low_detail

        self.names = ["O4", "C3", "C2", "C10", "C11", "C12", "C13", "C15", "C16",
                      "C17", "C18", "C20", "C21", "C22", "C23", "C24", "C26", "C27",
                      "C28", "C29", "C31", "C32", "C33", "C41", "C39", "O40"]

        try:
            experimental_data = pd.read_csv(f"data/{self.basename}_curve.csv")
            print("Curve data found, skipping re-calculating")
        except FileNotFoundError:
            try:
                print("Trying to calculate curve parameters from MD file...")
                create_curve(mdfile)
                experimental_data = pd.read_csv(f"data/{self.basename}_curve.csv")
            except FileNotFoundError:
                print("No MD data provided")
                sys.exit(1)

        self.experimental_potentials = np.array(experimental_data["mean"])
        self.upper = np.array(experimental_data["upper"])
        self.lower = np.array(experimental_data["lower"])

        self.experimental_derivative = discrete_derivative(self.experimental_potentials)

        try:
            self.vdwpoints = np.array(pd.read_csv(f"vdwpoints/vdwpoints_{density}.csv"))
        except FileNotFoundError:
            print("No such density file present in directory. Check directory for available variant")
            sys.exit(1)

        try:
            self.start_charges = np.array(pd.read_csv(f"vdwpoints/distances_{density}.csv")["charge"])
            self.distances_df = pd.read_csv(f"vdwpoints/distances_{density}.csv").drop(["charge", "Unnamed: 0"], axis=1)
            self.distances_df_inv = (1 / (self.distances_df * 1.889725988579)).T  # converted to bohrs
        except FileNotFoundError:
            print("No such density file present in directory. Check directory for available variant")
            sys.exit(1)

        print(f"Loaded {len(self.vdwpoints)} points")

        self.max_charge = max_charge

    def evaluate_potentials(self, charges: np.ndarray) -> np.ndarray:
        """
        Method for evaluating potentials using a distances table
        :param charges: charges (array of charge values)
        :return: new potentials array (26 points)
        """
        a = []

        for i in self.names:
            a1 = charges * self.distances_df_inv
            a.append(a1.T[i].sum())
        return np.array(a)

    def fitness(self, individual) -> (float, float,):
        """
        Method for fitness evaluation
        Uses two metrics so far: absolute values similarity and derivative similarity
        :param individual: individual from genetic algorithm
        :return: tuple of fitness values
        """

        new_pots = self.evaluate_potentials(individual)
        new_deriv = np.array(discrete_derivative(new_pots))

        # absolute similarity
        value_fitness_std = 0
        for v in range(len(new_pots)):

            # lower importance of this parameter if point reached std deviation interval
            if self.upper[v] > v > self.lower[v]:
                value_fitness_std += (((new_pots[v] - self.experimental_potentials[
                    v]) / self.experimental_potentials.mean()) ** 2) / 100  # normalized

            # if point doesn't reach std deviation interval the importance is higher
            else:
                value_fitness_std += (((new_pots[v] - self.experimental_potentials[
                    v]) / self.experimental_potentials.mean()) ** 2)  # normalized

        # derivative similarity
        deriv_fitness = sum(
            ((new_deriv - self.experimental_derivative) / self.experimental_derivative.mean()) ** 2)  # normalized

        return (value_fitness_std,
                deriv_fitness,)  # tuple

    def plot(self, ax: plt.axes, ax3d: plt.axes, best: np.ndarray, other: np.ndarray, charges: np.ndarray) -> None:
        """
        GUI creation
        :param ax: plt axes
        :param ax3d: plt 3d axes
        :param best: potentials of the best individual (or any other you want to plot)
        :param other: numpy 2d array with other individuals
        :param charges: charges of the best individual (or any other you want to plot)
        :return:
        """

        # target plot with stuff
        ax.plot(np.arange(26), self.experimental_potentials, label="experimental")
        ax.fill_between(np.arange(26), self.lower, self.upper, color='lightblue', alpha=.25,
                        label="std. deviation")  # standard deviation

        # derivative plot, uncomment if needed
        # ax.plot(np.arange(25), self.experimental_derivative, c="green")
        # ax.plot(np.arange(25), discrete_derivative(best), c="red")

        if not self.low_detail:
            # other individuals (a lot of grey lines)
            for i in other:
                ax.plot(np.arange(26), i, alpha=.01, color="grey")

        # best individual (orange line)
        ax.plot(np.arange(26), best, label="predicted")

        # plot points in 3d
        colored_points = pd.DataFrame(self.vdwpoints)
        colored_points["charges"] = charges
        if self.low_detail:
            colored_points = colored_points[colored_points["charges"] != 0]

        normalize = matplotlib.colors.Normalize(vmin=-self.max_charge, vmax=self.max_charge)

        ax3d.scatter(colored_points[0], colored_points[1], colored_points[2], marker='.', s=1,
                     c=colored_points["charges"], cmap="bwr", norm=normalize)

        sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(-self.max_charge * 2, self.max_charge * 2))
        labels = np.linspace(-self.max_charge*2, self.max_charge*2, 9).round(8)

        if not self.cb:
            cbar = plt.colorbar(sm, ax=ax3d, boundaries=np.linspace(-self.max_charge * 2, self.max_charge * 2, 1024))
            cbar.set_ticks(labels)
            cbar.set_ticklabels(labels)
            self.cb = True

        ax.legend(loc="lower left")
        ax.set_ylim(-0.25, 0.05)
        ax3d.set_xlim(-15, 15)
        ax3d.set_ylim(-15, 15)
        ax3d.set_zlim(-15, 15)

        ax.set_title(f"Generation {self.generation}")
        ax3d.set_title(f"{len(colored_points[colored_points['charges'] != 0])}/{len(self.vdwpoints)} points charged")

        if self.mkdir:
            plt.savefig(f"{self.working_dir}/figures/gen{self.generation:04d}.png", dpi=500)

    def genetic_algorithm(self, population_size, p_crossover, p_mutation,
                          max_generations, tournsize, hof_size):
        """
        Genetic algorithm itself
        :param population_size:
        :param p_crossover:
        :param p_mutation:
        :param max_generations:
        :param tournsize:
        :param hof_size:
        :return:
        """

        start_time = time.time()
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d%H%M%S")

        if self.mkdir:
            self.working_dir = (f"output/{date_time_str}_{len(self.vdwpoints)}_points_{population_size}_"
                                f"inds_{max_generations}_gens_{p_mutation}_pm_{p_crossover}_pc")
            os.mkdir(self.working_dir)
            os.mkdir(f"{self.working_dir}/figures")

        npoints = len(self.vdwpoints)  # number of points
        hof = tools.HallOfFame(maxsize=hof_size)

        creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))  # fitness weights
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        # toolbox.register("zeroValue", random.randint, 0, 0)  # fill all genes with zero values
        # toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroValue, npoints)

        # Initial charges
        def known_ind():
            return creator.Individual(self.start_charges)

        # toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
        toolbox.register("populationCreator", tools.initRepeat, list, known_ind)  # read initial individuals from file

        population = toolbox.populationCreator(n=population_size)

        toolbox.register("evaluate", self.fitness)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", mut_gaussian_degr, mu=0, sigma=self.max_charge,
                         indpb=2.0 / npoints)  # mutation freq - approx. 2 mutations per genome

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("best", np.min)
        stats.register("avg", np.mean)

        def show(ax, ax3d):
            """
            Plot update function (show every generation)
            :param ax: axes
            :param ax3d: 3d axes
            :return:
            """

            ax.clear()
            ax3d.clear()

            try:
                new_best = self.evaluate_potentials(hof.items[0])
                charges = np.array(hof.items[0])
            except IndexError:
                new_best = self.evaluate_potentials(self.start_charges)
                charges = self.start_charges

            other = []
            if not self.low_detail:
                # Before multiprocessing this part took even more time than the whole genetic algorithm...
                # it calculates the whole population potentials
                with multiprocessing.Pool() as pool:
                    for result in pool.map(self.evaluate_potentials, population):
                        other.append(result)
                other = np.array(other)

            self.plot(ax, ax3d, new_best, other, charges)
            self.generation += 1

            plt.draw()
            plt.gcf().canvas.flush_events()

        plt.ion()
        fig = plt.figure(figsize=(16.0, 6.0))
        ax = fig.add_axes((0.075, 0.1, 0.45, 0.8))
        ax3d = fig.add_axes((0.525, 0.1, 0.45, 0.8), projection='3d')

        show(ax, ax3d)

        # Genetic algorithm itself
        population, logbook = ea_simple_elitism(population, toolbox,
                                                cxpb=p_crossover,
                                                mutpb=p_mutation,
                                                ngen=max_generations,
                                                halloffame=hof,
                                                stats=stats,
                                                verbose=True,
                                                callback=(show, (ax, ax3d,)),
                                                )

        maxFitnessValues, meanFitnessValues = logbook.select("best", "avg")

        print(f"\nSimulation finished in {round((time.time() - start_time))} seconds\n")

        best = list(hof.items[0])
        print(f"Best charges:\n{best}")

        # Video rendering from frames
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

        charged_points = pd.DataFrame(self.vdwpoints)
        distances_charged = self.distances_df

        charged_points["charge"] = best
        # charged_points_only = charged_points[charged_points["charge"] != 0]
        distances_charged["charge"] = best
        # distances_charged_only = distances_charged[distances_charged["charge"] != 0]

        # saving logs and results
        if self.mkdir:
            charged_points.to_csv(f'{self.working_dir}/charged_points.csv', index=False)
            distances_charged.to_csv(f'{self.working_dir}/distances_charged.csv', index=False)
            # charged_points_only.to_csv(f'{self.working_dir}/charged_points_only.csv', index=False)
            # distances_charged_only.to_csv(f'{self.working_dir}/distances_charged_only.csv', index=False)
            with open(f"{self.working_dir}/data.txt", "w") as text_file:
                text_file.write(f"Job started: {now.strftime('%Y %m %d at %H:%M:%S')}\n"
                                f"Completed in {round(time.time() - start_time)} seconds\n\n"
                                f"npoints = {len(self.vdwpoints)}\n"
                                f"max_charge = {self.max_charge}\n"
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
