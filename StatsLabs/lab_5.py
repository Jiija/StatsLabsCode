import Distributions.distribution as ds
from matplotlib import pyplot as plt

distributions = [ds.Normal2D(0, 0, 1, 1, 0), ds.Normal2D(0, 0, 1, 1, 0.5), ds.Mixed([ds.Normal2D(0, 0, 1, 1, 0.9),
                                                                                    ds.Normal2D(0, 0, 10, 10, -0.9)], [0.9, 0.1])]

table_names = ["RHO=0", "RHO=0.5", "MIXED"]
dataset_sizes = [20, 60, 100]
n_trials = 1000

for i, ds in enumerate(distributions):
    f = open("Reports/Lab5/Tables/" + table_names[i] + ".tex", 'w')
    print(ds.calculate_correlation_coeffs(dataset_sizes, n_trials), file=f)
    f.close()

for i in range(len(distributions) - 1):
    for size in dataset_sizes:
        distributions[i].plot_ellipse(size)
        plt.savefig("Reports/Lab5/Figures/ellpise_rho =" + str(distributions[i]._rho) + "_n = " + str(size) + ".png")
        plt.close()

