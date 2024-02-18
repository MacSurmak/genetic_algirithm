"""
Separate module for points file generation. It requires pyvdwsurface installed

https://github.com/rmcgibbo/pyvdwsurface
"""

import numpy as np
import pandas as pd

from pyvdwsurface import vdwsurface
from scipy.spatial import distance


def generate(filename="can.xyz", density=1) -> None:
    """
    Generates points on a VDW surface with given density
    :param filename: molecule file name
    :param density: points mesh density
    :return: None
    """
    elements = np.loadtxt(filename, usecols=0, dtype=bytes)
    xs = np.loadtxt(filename, usecols=1, dtype=float)
    ys = np.loadtxt(filename, usecols=2, dtype=float)
    zs = np.loadtxt(filename, usecols=3, dtype=float)
    atoms = []
    for i in range(len(elements)):
        x = xs[i]
        y = ys[i]
        z = zs[i]
        atoms.append([x, y, z])
    atoms = np.array(atoms)
    vdwpoints = pd.DataFrame(vdwsurface(atoms, elements, density=density))
    vdwpoints.to_csv(f'vdwpoints/vdwpoints_{density}.csv', index=False)

    chain_coordinates = []
    for i in [14, 13, 11, 10, 9, 8, 6, 5, 4, 3, 2, 20, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37]:
        chain_coordinates.append(atoms[i - 1])

    names = ["O4", "C3", "C2", "C10", "C11", "C12", "C13", "C15", "C16",
             "C17", "C18", "C20", "C21", "C22", "C23", "C24", "C26", "C27",
             "C28", "C29", "C31", "C32", "C33", "C41", "C39", "O40"]
    distances_df = pd.DataFrame()

    counter = 0
    for atom in np.array(chain_coordinates):
        dists = []
        for point in np.array(vdwpoints):
            dist = distance.euclidean(atom, point)
            dists.append(dist)
        distances_df[f"{names[counter]}"] = dists
        counter += 1

    distances_df["charge"] = np.zeros(len(distances_df))
    distances_df.to_csv(f"vdwpoints/distances_{density}.csv")

generate(density=50)
