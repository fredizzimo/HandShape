import numpy as np
import scipy as sp
import scipy.stats as stats
import itertools

class Shade:
    def __init__(self, f, bounds, max_fevals, population_size, p, archive_rate, memory_size, learning_rate):
        self.bounds = np.array(list(bounds), ndmin=2)
        self.num_dim = len(self.bounds)
        self.f = f
        self.initial_population_size = population_size
        self.p = p
        self.archive_rate = archive_rate
        self.memory_size = memory_size
        self.max_fevals = max_fevals
        self.learning_rate = learning_rate

    def optimize(self):
        runtime = Shade.Runtime(self)
        runtime.optimize()
        return runtime.best

    class Runtime:
        def __init__(self, parent):
            self.bounds = parent.bounds
            self.num_dim = parent.num_dim
            self.f = parent.f
            self.initial_population_size = parent.initial_population_size
            self.p = parent.p
            self.archive_rate = parent.archive_rate
            self.memory_size = parent.memory_size
            self.max_fevals = parent.max_fevals
            self.learning_rate = parent.learning_rate

            self.nevals = 0
            self.best = np.finfo(np.float64).max, np.empty(self.num_dim,)
            self.memory_sf = np.full(self.memory_size, 0.5)
            self.memory_cr = np.full(self.memory_size, 0.5)
            self.memory_er = np.full(self.memory_size, 0.5)
            self.generation = 0
            self.population_size = self.initial_population_size
            self.population = np.random.random((self.population_size, self.num_dim))
            self.population_values = self.evaluate_population(self.population)
            self.covariance = np.cov(self.population, rowvar=False)
            self.archive_size = int(self.population_size * self.archive_rate)
            self.archive = np.empty((self.archive_size, self.num_dim))
            self.current_archive_size = 0
            self.p_max = int(self.population_size * self.p)
            self.memory_pos = 0

        def optimize(self):

            while True:
                self.sorted_population = np.argsort(self.population_values)
                best_index = self.sorted_population[0]
                if self.population_values[best_index] < self.best[0]:
                    self.best = self.population_values[best_index], self.scale(self.population[best_index])

                new_population_size = \
                    round(((4 - self.initial_population_size) / self.max_fevals) * self.nevals + self.initial_population_size)

                if new_population_size < self.population_size:
                    self.population = self.population[self.sorted_population][:new_population_size].copy()
                    self.population_values = self.population_values[self.sorted_population][:new_population_size].copy()
                    self.sorted_population = np.arange(new_population_size)
                    self.population_size = new_population_size
                    self.archive_size = int(self.population_size * self.archive_rate)
                    if self.current_archive_size > self.archive_size:
                        self.current_archive_size = self.archive_size
                    self.p_max = int(self.population_size * self.p)
                    self.covariance = np.cov(self.population, rowvar=False)

                print("New population size:", new_population_size)
                print("Generation %i, evals %i, f: %f" % (self.generation, self.nevals, self.best[0]))
                if self.nevals >= self.max_fevals:
                    break
                h_indices = np.random.randint(0, self.memory_size, self.population_size)
                mu_sf = self.memory_sf[h_indices]
                mu_cr = self.memory_cr[h_indices]
                mu_er = self.memory_er[h_indices]

                def gaussian(mu):
                    if (mu != 0):
                        return np.clip(np.random.normal(mu, 0.1), 0.0, 1.0)
                    else:
                        return 0
                pop_cr = np.fromiter((gaussian(mu) for mu in mu_cr), dtype=np.float64, count=self.population_size)
                pop_er = np.fromiter((gaussian(mu) for mu in mu_er), dtype=np.float64, count=self.population_size)

                def cauchy(mu):
                    ret = -0.1
                    while ret <= 0.0:
                        ret = stats.cauchy.rvs(mu, 0.1)
                    if ret > 1.0:
                        ret = 1.0
                    return ret

                pop_sf = np.fromiter((cauchy(mu) for mu in mu_sf), dtype=np.float64, count=self.population_size)

                new_pop = np.fromiter(itertools.chain.from_iterable(
                    (self.currentToPBest1Bin(i, pop_cr[i], pop_sf[i], pop_er[i])
                        for i in range(self.population_size))),
                    dtype=np.float64, count=self.population_size * self.num_dim)
                new_pop.shape = (self.population_size, self.num_dim)

                new_pop_values = self.evaluate_population(new_pop)

                success_indices = []

                deltas = new_pop - self.population

                for i in range(self.population_size):
                    if new_pop_values[i] <= self.population_values[i]:
                        if new_pop_values[i] < self.population_values[i]:
                            if self.current_archive_size == self.archive_size:
                                archive_index =np.random.randint(0, self.archive_size)
                            else:
                                archive_index = self.current_archive_size
                                self.current_archive_size += 1
                            self.archive[archive_index] = self.population[i]

                            self.population_values[i] = new_pop_values[i]
                            success_indices.append(i)
                        self.population[i] = new_pop[i]

                if len(success_indices):
                    success_sf = pop_sf[success_indices]
                    success_cr = pop_cr[success_indices]
                    success_er = pop_er[success_indices]
                    success_deltas = deltas[success_indices]
                    success_distances = np.linalg.norm(success_deltas, axis=1)

                    deltasum = np.sum(success_distances)
                    weights = success_distances / deltasum

                    newsf = np.sum(weights * success_sf * success_sf)
                    if newsf != 0:
                        newsf /= np.sum(weights * success_sf)

                    newcr = np.sum(weights * success_cr * success_cr)
                    if newcr != 0:
                        newcr /= np.sum(weights * success_cr)

                    newer = np.sum(weights * success_er * success_er)
                    if newer != 0:
                        newer /= np.sum(weights * success_er)

                    self.memory_sf[self.memory_pos % self.memory_size] = newsf
                    self.memory_er[self.memory_pos % self.memory_size] = newer
                    if self.memory_cr[self.memory_pos % self.memory_size] != 0:
                        self.memory_cr[self.memory_pos % self.memory_size] = newcr
                    self.memory_pos += 1

                    print("Parameter adaptation sf: %f, cr: %f, er: %f" % (newsf, newcr, newer))

                self.generation += 1

        def evaluate_population(self, population):
            iter = (self.f(self.scale(args)) for args in population)
            self.nevals += len(population)
            ret = np.fromiter(iter, dtype=np.float64, count=len(population))
            return ret

        def scale(self, args):
            iter = (args[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0] for i in range(self.num_dim))
            return np.fromiter(iter, dtype=np.float64, count=self.num_dim)

        def currentToPBest1Bin(self, current, cross_rate, scaling_factor, eigen_ratio):
            if self.p_max > 0:
                pbesti = np.random.randint(0, self.p_max)
            else:
                pbesti = 0
            pbesti = self.sorted_population[pbesti]
            r1 = current
            r2 = current
            while r1 == current:
                r1 = np.random.randint(0, self.population_size)
            while r2 == current or r2 == r1:
                r2 = np.random.randint(0, self.population_size + self.current_archive_size)
            if r2 < self.population_size:
                r2_value = self.population[r2]
            else:
                r2_value = self.archive[r2 - self.population_size]

            donor = np.fromiter((
                self.population[current][i] + scaling_factor *
                (self.population[pbesti][i] - self.population[current][i]) +
                scaling_factor * (self.population[r1][i] - r2_value[i]) for i in range(self.num_dim)),
                dtype=np.float64, count=self.num_dim)

            cw = self.learning_rate * (1.0 - self.nevals / self.max_fevals)
            self.covariance = (1 - cw) * self.covariance + cw * np.cov(self.population, rowvar=False)
            _, b = np.linalg.eig(self.covariance)
            b_conjugate_transpose = b.conj().T

            eig = np.random.random() < eigen_ratio

            if eig:
                p = np.dot(b_conjugate_transpose, self.population[current])
                v = np.dot(b_conjugate_transpose, donor)
            else:
                p = self.population[current]
                v = donor

            random_variable = np.random.randint(0, self.num_dim)
            xover = (v[i] if np.random.random() < cross_rate or i == random_variable else p[i] for i in range(self.num_dim))
            ret = np.fromiter(xover, np.float64, count=self.num_dim)

            if eig:
                ret = np.dot(b, ret)

            for i in range(self.num_dim):
                # Fixup bounds
                if ret[i] < 0.0:
                    ret[i] = self.population[current][i] * 0.5
                elif ret[i] > 1.0:
                    ret[i] = (1.0 - self.population[current][i]) * 0.5

            return ret
