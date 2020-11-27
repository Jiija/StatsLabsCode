import numpy as np
from scipy import stats
from scipy import optimize
from matplotlib import pyplot as plt


def quadrantr(x, y):
    return np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))


def lstmod(x, y):
    M = lambda v: np.sum(np.abs(y - (v[0] + v[1] * x)))
    res = optimize.minimize(M, np.array([0, 0]), method='Nelder-Mead').x
    a = res[0]
    b = res[1]
    return a, b


bounds = [-1.8, 2]
num_points = 20
coeffs = [2, 2]

x = np.linspace(bounds[0], bounds[1], num_points)
y0 = coeffs[0] + coeffs[1] * x
y1 = y0 + stats.norm(0, 1).rvs(num_points)
y2 = np.copy(y1)
y2[0] += 10
y2[-1] -= 10

regressions = [
    {
        'type': "simple",
        'y': y1
    },
    {
        'type': "robust",
        'y': y2
    }
]

for r in regressions:
    y = r['y']
    least_sq_b, least_sq_a = stats.linregress(x, y)[0:2]
    least_mod_a, least_mod_b = lstmod(x, y)
    print(r['type'])
    print("least_square:", "a =", round(least_sq_a, 2),
          "b =", round(least_sq_b, 2), "Q =", np.round(np.sum(np.square(y - (least_sq_a + least_sq_b * x))), 4),
          "M =", np.round(np.sum(np.abs(y - (least_sq_a + least_sq_b * x))), 4))
    print("least_module:", "a =", round(least_mod_a, 2),
          "b =", round(least_mod_b, 2), "Q =", np.round(np.sum(np.square(y - (least_mod_a + least_mod_b * x))), 4),
          "M =", np.round(np.sum(np.abs(y - (least_mod_a + least_mod_b * x))), 4))
    print()

    plt.plot(x, y0, 'gray', label="Модель")
    plt.plot(x, least_sq_a + least_sq_b * x, 'darkcyan', label="МНК")
    plt.plot(x, least_mod_a + least_mod_b * x, 'red', label="МНМ")
    plt.scatter(x, y, facecolors='none', edgecolors='black', zorder=100, label="Выборка")
    plt.legend()
    plt.savefig("Reports/Lab6/Figures/" + r['type'] + ".png")
    plt.close()
