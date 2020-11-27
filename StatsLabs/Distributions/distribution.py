from abc import ABC, abstractmethod
import math
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import Util.tex_table as tt

shift = 10 ** (-7)


class Distribution(ABC):
    def __init__(self):
        self._distr = stats.uniform()
        self._left_bound = 0
        self._right_bound = 0
        self._lower_bound = 0
        self._upper_bound = 0
        self.dist_color = 'green'
        self.hist_color = 'lightgreen'
        self.boxplot_vert = False
        self.boxplot_medianprops = dict(color='g')
        self._a = []
        self._b = []

    def build_boxplots(self, dataset_sizes):
        data = [self._generate_hist_data(size) for size in dataset_sizes]

        fig, ax = plt.subplots()
        plt.grid(axis='x')
        ax.boxplot(data, vert=self.boxplot_vert, medianprops=self.boxplot_medianprops, labels=["N = " + str(size)
                                                                                               for size in dataset_sizes])

    def outliers_rate(self, dataset_size, num_tests, k):
        outliers = np.array([self._outliers(dataset_size, k) / dataset_size for i in range(num_tests)])

        rate = sum(outliers) / num_tests
        variance = sum(outliers ** 2) / num_tests - rate ** 2

        return rate, variance

    def _outliers(self, dataset_size, k):
        data = self._generate_hist_data(dataset_size)

        l_q = np.quantile(data, 0.25)
        u_q = np.quantile(data, 0.75)
        i_q_r = u_q - l_q

        left_bound = l_q - k * i_q_r
        right_bound = u_q + k * i_q_r

        return sum(1 for value in data if value < left_bound or value > right_bound)

    def theoretical_outliers_prob(self, k):
        range = self._distr.ppf(0.75) - self._distr.ppf(0.25)
        x_1 = self._distr.ppf(0.25) - k * range
        x_2 = self._distr.ppf(0.75) + k * range
        return self._distr.cdf(x_1) - (self._distr.cdf(x_1 + shift) - self._distr.cdf(x_1 - shift)) + 1 - self._distr.cdf(x_2)

    def plot_ecdf(self, dataset_sizes):
        class ECDF:
            data = []

            def __init__(self, data):
                self.data = data
                self.evaluate_vector = np.vectorize(self.evaluate)

            def evaluate(self, x):
                count = self.data[self.data < x]
                return count.size / self.data.size

        n = len(dataset_sizes)
        plt.subplots(n)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.xlim(self._a, self._b)
            plt.xlabel('x')
            plt.ylabel('ECDF')

            sample = self._generate_hist_data(dataset_sizes[i])
            ecdf = ECDF(sample)
            x = np.linspace(self._a, self._b)
            y = ecdf.evaluate_vector(x)
            y_cdf = self._distr.cdf(x)

            plt.plot(x, y, color=self.dist_color)
            plt.plot(x, y_cdf, color=self.hist_color)
            plt.title('N = ' + str(dataset_sizes[i]))

    def plot_kde(self, dataset_sizes, hor_scale, str_hor_scale, str_num):
        fig, ax = plt.subplots(nrows=len(dataset_sizes), ncols=len(hor_scale), sharex=True, sharey=True, figsize=(8, 8))
        samples = [self._generate_hist_data(size) for size in dataset_sizes]

        k = 0
        for row in ax:
            i = 0
            for col in row:
                kde = stats.gaussian_kde(samples[i], bw_method='silverman')
                col.set_xlim(self._a, self._b)
                col.set_xlabel('x')
                col.set_ylabel('KDE')

                x = np.linspace(self._a, self._b)
                y_pdf = self._generate_dist_data(x)

                kde.set_bandwidth(bw_method=kde.factor * hor_scale[k])
                y_kde = kde.evaluate(x)

                col.plot(x, y_kde, color=self.dist_color)
                col.plot(x, y_pdf, color=self.hist_color)
                col.set_title(str_num + str(dataset_sizes[i]) + ', ' + str_hor_scale[k])
                i += 1
            k += 1
            fig.tight_layout(pad=1.0)

    @abstractmethod
    def _generate_dist_data(self, x):
        pass

    @abstractmethod
    def _generate_hist_data(self, dataset_size):
        pass

    def calculate_characteristics(self, num_trials, dataset_sizes):
        tex = tt.heading(" &$\\bar{x}$& $med x$ & $ z_{R}$ & $z_{Q}$ & $z_{tr}$")
        for size in dataset_sizes:
            data = [self._generate_hist_data(size) for i in range(num_trials)]
            means = np.array([np.mean(d) for d in data])
            medians = np.array([np.median(d) for d in data])
            z_Rs = np.array([(np.max(d) + np.min(d)) / 2 for d in data])
            z_Qs = np.array([(np.quantile(d, 0.25) + np.quantile(d, 0.75)) / 2 for d in data])
            z_trs = np.array([stats.trim_mean(d, 0.25) for d in data])

            mean_mean = np.mean(means)
            median_mean = np.mean(medians)
            zR_mean = np.mean(z_Rs)
            zQ_mean = np.mean(z_Qs)
            ztr_mean = np.mean(z_trs)

            mean_var = np.var(means)
            median_var = np.var(medians)
            zR_var = np.var(z_Rs)
            zQ_var = np.var(z_Qs)
            ztr_var = np.var(z_trs)

            tex += tt.empty_row("$N=" + str(size) + "$", 5)
            tex += tt.row("$\\mathbf{E}\\left[z\\right]$", [
                np.round(mean_mean, 6),
                np.round(median_mean, 6),
                np.round(zR_mean, 6),
                np.round(zQ_mean, 6),
                np.round(ztr_mean, 6)
            ])

            tex += tt.row("$\\mathbf{D}\\left[z\\right]$", [
                np.round(mean_var, 6),
                np.round(median_var, 6),
                np.round(zR_var, 6),
                np.round(zQ_var, 6),
                np.round(ztr_var, 6)
            ])

        return tt.tabular(tex, 6)

    def chi_sq(self, sample_size, alpha):
        sample = self._distr.rvs(sample_size)
        k = math.ceil(1.72 * sample_size ** (1 / 3))
        chi_sq = stats.chi2(df=k - 1)

        a = []
        centre = self._distr.ppf(0.5)
        h = 2 * stats.iqr(sample) / (sample_size ** (1 / 3))

        a.append(-np.inf)
        for i in range(k - 1):
            a.append(centre + h * (i - (k - 1) / 2))
        a.append(np.inf)

        p = np.zeros(k)
        n = np.zeros(k)

        for i in range(k):
            p[i] = self._distr.cdf(a[i + 1]) - self._distr(a[i])
            y = sample[a[i] < sample]
            n[i] = len(y[y <= a[i + 1]])

        chi_sq_1 = (1 / sample_size) * sum(np.square(n - sample_size * p) / p)
        chi_sq_2 = chi_sq.ppf(1 - alpha)

        return chi_sq_1 < chi_sq_2, np.round([chi_sq_1, chi_sq_2], 4), np.round(a, 2), np.round(p, 4), n, \
               np.round(n - sample_size * p, 4)


class ContinuousDistribution(Distribution):
    def __init__(self):
        super(ContinuousDistribution, self).__init__()
        self._a = -4.0
        self._b = 4.0

    def build_plots(self, dataset_size, points_density):
        number_of_bins = math.ceil(dataset_size ** (1 / 2))
        hist_vals = self._generate_hist_data(dataset_size)

        x_range = self._right_bound - self._left_bound
        bin_width = x_range / number_of_bins

        plt.xlim(self._left_bound, self._right_bound)
        plt.ylim(self._lower_bound, self._upper_bound)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title("N = " + str(dataset_size))
        plt.grid()

        x = np.linspace(self._left_bound, self._right_bound, int(x_range * points_density))
        y = self._generate_dist_data(x)
        plt.plot(x, y, color=self.dist_color)

        bins = np.arange(min(hist_vals), max(hist_vals) + bin_width, bin_width)
        plt.hist(x=hist_vals, bins=bins, color=self.hist_color, density=True)


class DiscreteDistribution(Distribution):
    def __init__(self):
        super(DiscreteDistribution, self).__init__()
        self._a = int(6)
        self._b = int(14)
        self.marker = 'o'

    def build_plots(self, dataset_size, step):
        hist_vals = self._generate_hist_data(dataset_size)

        plt.xlim(self._left_bound, self._right_bound)
        plt.ylim(self._lower_bound, self._upper_bound)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title("N = " + str(dataset_size))
        plt.grid()

        x = np.arange(self._left_bound, self._right_bound, step)
        y = self._generate_dist_data(x)
        plt.plot(x, y, color=self.dist_color, marker=self.marker)

        bins = np.arange(min(hist_vals), max(hist_vals), step)
        plt.hist(x=hist_vals, bins=bins, color=self.hist_color, density=True)


class Poisson(DiscreteDistribution):
    def __init__(self, lambda_=10):
        super(Poisson, self).__init__()
        self._distr = stats.poisson(lambda_)
        self._lambda = lambda_
        self._left_bound = 0
        self._right_bound = 3 * lambda_
        self._lower_bound = 0
        self._upper_bound = 2 / ((2 * math.pi * lambda_) ** (1 / 2))

    def _generate_dist_data(self, x):
        return stats.poisson.pmf(np.ceil(x), self._lambda)

    def _generate_hist_data(self, dataset_size):
        return stats.poisson.rvs(self._lambda, size=dataset_size)


class Cauchy(ContinuousDistribution):
    def __init__(self, loc=0, scale=1):
        super(Cauchy, self).__init__()
        self._distr = stats.cauchy(loc=loc, scale=scale)
        self._loc = loc
        self._scale = scale
        self._lower_bound = 0
        self._upper_bound = 1.5 / (math.pi * self._scale)

    def _generate_hist_data(self, dataset_size):
        values = stats.cauchy.rvs(loc=self._loc, scale=self._scale, size=dataset_size)
        self._left_bound = min(values)
        self._right_bound = max(values)
        return values

    def _generate_dist_data(self, x):
        return stats.cauchy.pdf(x, loc=self._loc, scale=self._scale)


class Laplace(ContinuousDistribution):
    def __init__(self, loc=0, scale=0.5 ** (1 / 2)):
        super(Laplace, self).__init__()
        self._loc = loc
        self._scale = scale
        sigma = (2 * (scale ** 2)) ** (1 / 2)
        self._distr = stats.laplace(loc=loc, scale=scale)
        self._left_bound = loc - 3 * sigma
        self._right_bound = loc + 3 * sigma
        self._lower_bound = 0
        self._upper_bound = scale

    def _generate_hist_data(self, dataset_size):
        return stats.laplace.rvs(loc=self._loc, scale=self._scale, size=dataset_size)

    def _generate_dist_data(self, x):
        return stats.laplace.pdf(x, loc=self._loc, scale=self._scale)


class Normal(ContinuousDistribution):
    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self._distr=stats.norm(mu, sigma)
        self._left_bound = mu - 3 * sigma
        self._right_bound = self._left_bound + 6 * sigma
        self._mu = mu
        self._sigma = sigma
        self._lower_bound = 0
        self._upper_bound = 0.6

    def _generate_dist_data(self, x):
        return stats.norm.pdf(x, self._mu, self._sigma)

    def _generate_hist_data(self, dataset_size):
        return stats.norm.rvs(self._mu, self._sigma, dataset_size)


class Uniform(ContinuousDistribution):
    def __init__(self, l=-(3 ** (1 / 2)), r=3 ** (1 / 2)):
        super(Uniform, self).__init__()
        self._distr = stats.uniform(l, r - l)
        self._l = l
        self._r = r
        self._left_bound = 1.5 * l
        self._right_bound = 1.5 * r
        self._lower_bound = 0
        self._upper_bound = 2 / (r - l)

    def _generate_hist_data(self, dataset_size):
        return stats.uniform.rvs(self._l, self._r - self._l, dataset_size)

    def _generate_dist_data(self, x):
        return stats.uniform.pdf(x, self._l, self._r - self._l)


class Distribution2D(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def rvs(self, size):
        pass

    def calculate_correlation_coeffs(self, dataset_sizes, n_trials):
        def digits_for_rounding(val):
            return max(0, round(-math.log10(abs(val))))

        tex = ""
        for size in dataset_sizes:
            tex += tt.heading("$N = " + str(size) + "$&$r$&$r_S$&$r_Q$")
            pearson = np.zeros(n_trials)
            spearman = np.zeros(n_trials)
            quadrant = np.zeros(n_trials)

            for i in range(n_trials):
                data = self.rvs(size)
                x = data[:, 0]
                y = data[:, 1]
                m_x = np.median(x)
                m_y = np.median(y)
                pearson[i], p = stats.pearsonr(x, y)
                spearman[i], p = stats.spearmanr(x, y)
                quadrant[i] = np.mean(np.sign(x - m_x) * np.sign(y - m_y))

            pearson_variance = np.var(pearson)
            spearman_variance = np.var(spearman)
            quadrant_variance = np.var(quadrant)

            tex += tt.row("$\\mathbf{E}\\left[z\\right]$",
                          [
                           np.round(np.mean(pearson), digits_for_rounding(pearson_variance)),
                           np.round(np.mean(spearman), digits_for_rounding(spearman_variance)),
                           np.round(np.mean(quadrant), digits_for_rounding(quadrant_variance))
                          ])

            tex += tt.row("$\\mathbf{E}\\left[z^2\\right]$",
                          [
                              np.round(np.mean(pearson ** 2), digits_for_rounding(pearson_variance)),
                              np.round(np.mean(spearman ** 2), digits_for_rounding(spearman_variance)),
                              np.round(np.mean(quadrant ** 2), digits_for_rounding(quadrant_variance))
                          ])

            tex += tt.row("$\\mathbf{D}\\left[z\\right]$",
                          [
                              np.round(np.mean(pearson_variance), 6),
                              np.round(np.mean(spearman_variance), 6),
                              np.round(np.mean(quadrant_variance), 6)
                          ])
            tex = tt.tabular(tex, 4)
            return tex


class Normal2D(Distribution2D):
    def __init__(self, loc_x, loc_y, scale_x, scale_y, rho):
        self._loc_x = loc_x
        self._loc_y = loc_y
        self._scale_x = scale_x
        self._scale_y = scale_y
        self._rho = rho

        cov_xy = rho * scale_x * scale_y
        disp_x = scale_x ** 2
        disp_y = scale_y ** 2

        self._distr = stats.multivariate_normal(mean=[loc_x, loc_y], cov=[[disp_x, cov_xy], [cov_xy, disp_y]])

    def pdf(self, x):
        return self._distr.pdf(x)

    def rvs(self, size):
        return self._distr.rvs(size=size)

    @property
    def mean_x(self):
        return self._loc_x

    @property
    def mean_y(self):
        return self._loc_y

    @property
    def sigma_x(self):
        return self._scale_x

    @property
    def sigma_y(self):
        return self._scale_y

    def plot_ellipse(self, size):
        class Ellipse:
            def __init__(self, mean_x, mean_y, sigma_x, sigma_y, rho):
                self._a = 1 / sigma_x ** 2
                self._b = 2 * rho / (sigma_x * sigma_y)
                self._c = 1 / sigma_y ** 2
                self._x0 = mean_x
                self._y0 = mean_y

            def z(self, x, y):
                x_diff = x - self._x0
                y_diff = y - self._y0
                return self._a * x_diff ** 2 - self._b * x_diff * y_diff + self._c * y_diff ** 2

            def rad2(self, samples):
                return max(self.z(samples[:, 0], samples[:, 1]))

        ellipse = Ellipse(self.mean_x, self.mean_y, self.sigma_x, self.sigma_y, self._rho)

        data = self.rvs(size)
        plt.scatter(data[:, 0], data[:, 1])

        plt.title("R = " + str(round(ellipse.rad2(data), 6)))
        x = np.linspace(min(data[:, 0]) - 2, max(data[:, 0]) + 2, 100)
        y = np.linspace(min(data[:, 1]) - 2, max(data[:, 1]) + 2, 100)
        x, y = np.meshgrid(x, y)
        z = ellipse.z(x, y)
        plt.contour(x, y, z, [ellipse.rad2(data)])


class Mixed(Distribution2D):
    def __init__(self, distributions, coeffs, dim=2):
        coeffs = np.array(coeffs) / sum(coeffs)

        self._distributions = distributions
        self._coeffs = coeffs
        self._n = len(distributions)
        self._dim = dim

    def pdf(self, x):
        return sum([self._distributions[i].pdf(x) * self.coeffs[i] for i in range(self._n)])

    def rvs(self, size):
        data = np.zeros((size, self._dim, self._n))
        for k, distribution in enumerate(self._distributions):
            data[:, k] = distribution.rvs(size=size)
        return data[np.arange(size), np.random.choice(np.arange(self._n), size=size, p=self._coeffs)]
