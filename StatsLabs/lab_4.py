import Distributions.distribution as ds
from matplotlib import pyplot as plt

distributions = [ds.Normal(), ds.Laplace(), ds.Uniform(l=-(3 ** (1 / 2)), r=(3 ** (1 / 2))), ds.Cauchy(), ds.Poisson()]
dataset_sizes = [20, 60, 100]
h_scale = [0.5, 1, 2]
str_h_scale = [r"$h = h_n / 2$", r"$h = h_n$", r"$h = 2h_n$"]
str_n = r"$N = $"

for ds in distributions:
    ds.plot_ecdf(dataset_sizes)
    plt.tight_layout(pad=1.0)
    plt.savefig('Reports/Lab4/Figures/ECDF' + ds.__class__.__name__ + '.png')

    ds.plot_kde(dataset_sizes, h_scale, str_h_scale, str_n)
    plt.savefig('Reports/Lab4/Figures/KDE' + ds.__class__.__name__ + '.png')
