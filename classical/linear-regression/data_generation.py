import csv
import numpy as np

no_samples = 5000
Xy_min, Xy_max = -100000, 100000

X = np.random.uniform(Xy_min, Xy_max, no_samples)
y = np.random.uniform(Xy_min, Xy_max, no_samples)

def func(X, y):
    return 3*X - 2*y + 5 + np.random.random(no_samples)

z = func(X, y)

with open("data/planar.csv", "w+") as f:
    writer = csv.writer(f)
    writer.writerow(["X", "y", "z"])
    writer.writerows(zip(X, y, z))