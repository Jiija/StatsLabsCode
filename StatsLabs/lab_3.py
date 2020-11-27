import Distributions.distribution as ds
from matplotlib import pyplot as plt

distributions = [ds.Normal(), ds.Laplace(), ds.Uniform(), ds.Cauchy(), ds.Poisson()]
dataset_sizes = [20, 100]

for ds in distributions:
    ds.build_boxplots(dataset_sizes)
    #plt.savefig('Reports/Lab3/Figures/' + ds.__class__.__name__ + '.png')
    plt.show()
    print("For " + ds.__class__.__name__ + " distribution:")
    for size in dataset_sizes:
        print("N = " + str(size) + ":")
        rate, variance = ds.outliers_rate(size, 1000, 1.5)
        print("rate = " + str(rate) + ", variance = " + str(variance))
    print()
    print("Theoretical outliers' probability: p = " + str(ds.theoretical_outliers_prob(1.5)))
    print()
    print()
