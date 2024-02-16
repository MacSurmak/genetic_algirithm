import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pyvdwsurface import vdwsurface

file_input = open("can.xyz",'r')
lines = file_input.readlines()
elements = []
coordinates = []
for line in lines:
    elem = line[:1].encode('utf-8')
    elements.append(elem)
    x = float(line[2:12])
    y = float(line[13:23])
    z = float(line[23:34])
    coordinates.append([x, y, z])

atoms = np.array(coordinates, dtype=float)
points = vdwsurface(atoms, elements, density=20)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], marker='o', s=0.5)
ax.set_xlim(-5,-35)
ax.set_ylim(70,100)
ax.set_zlim(30,60)
plt.savefig('can.png')
plt.show()
