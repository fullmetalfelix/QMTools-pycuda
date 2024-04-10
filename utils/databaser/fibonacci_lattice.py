#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

def fibonacci_sphere(n_points, offset):

    t = np.arange(n_points) + offset

    x0, _ = np.modf(t / GOLDEN_RATIO)
    y0 = (t / n_points) % 1
    print(x0)

    theta = 2 * np.pi * x0
    phi = np.arccos(1 - 2 * y0)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    xyz = np.array([x, y, z]).T

    return xyz

dr = 3
nr = 6
rs = dr * np.arange(1, nr+1)
n_points = 4

points = []
c = []
for i, r0 in enumerate(rs):
    offset = 2 * i / nr
    r = r0 + np.linspace(-0.5 * dr, 0.5 * dr, n_points)
    xyz = fibonacci_sphere(n_points, offset)
    points.append(r[:, None] * xyz)
    c.append([i] * n_points)

points = np.concatenate(points, axis=0)
c = np.concatenate(c)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)
ax.axis("equal")
# plt.scatter(x, y, c=t)
plt.show()
