import Distributions.distribution as ds
from matplotlib import pyplot as plt
from numpy import random

random.seed(198)
data_sizes = [10, 50, 1000]
num_of_plots = len(data_sizes)
points_density = 200

continuous_distributions = [ds.Normal(), ds.Laplace(), ds.Uniform(), ds.Cauchy()]


for distribution in continuous_distributions:
    plt.clf()
    fig, axs = plt.subplots(num_of_plots)

    for i in range(num_of_plots):
        plt.subplot(1, num_of_plots, i + 1)
        distribution.build_plots(data_sizes[i], points_density)
    fig.suptitle(distribution.__class__.__name__ + " distribution")

    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('Reports/Lab1/Figures/' + distribution.__class__.__name__ + '.png')

distribution = ds.Poisson()
plt.clf()
fig, axs = plt.subplots(num_of_plots)
for i in range(num_of_plots):
    plt.subplot(1, num_of_plots, i + 1)
    distribution.build_plots(data_sizes[i], 1)
fig.suptitle(distribution.__class__.__name__ + " distribution")

fig.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('Reports/Lab1/Figures/' + distribution.__class__.__name__ + '.png')
