from Distributions import distribution as ds
dataset_sizes = [10, 100, 10000]
distributions = [ds.Normal(), ds.Laplace(), ds.Uniform(), ds.Cauchy(), ds.Poisson()]
f = open("Reports/Lab2/tables.tex", 'w')

for ds in distributions:
    table = ds.calculate_characteristics(1000, dataset_sizes)
    print(table, file=f)
    print('', file=f)

f.close()
