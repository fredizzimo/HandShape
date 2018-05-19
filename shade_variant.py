import numpy as np
import scipy as sp
import scipy.stats as stats
import itertools

class Shade:
    def __init__(self, f, bounds, max_fevals, population_size, archive_size, history_size):
        self.bounds = np.array(list(bounds), ndmin=2)
        self.num_dim = len(self.bounds)
        self.f = f
        self.population_size = population_size
        self.archive_size = archive_size
        self.history_size = history_size
        self.max_fevals = max_fevals


    def optimize(self):
        self.nevals = 0
        self.best = np.finfo(np.float64).max, np.empty(self.num_dim,)
        memory_f = np.full(self.history_size, 0.5)
        memory_cr = np.full(self.history_size, 0.5)
        generation = 0
        self.population = np.random.random((self.population_size, self.num_dim))
        population_values = self.evaluate_population(self.population)
        p_max = int(self.population_size * 0.2)
        while True:
            print("Generation %i, evals %i, f: %f" % (generation, self.nevals, self.best[0]))
            if self.nevals >= self.max_fevals:
                break
            h_indices = np.random.randint(0, self.history_size, self.population_size)
            mu_sf = memory_f[h_indices];
            mu_cr = memory_cr[h_indices];

            pop_cr = np.random.normal(mu_cr, 0.1, self.population_size)
            pop_cr = np.clip(pop_cr, 0.0, 1.0)

            def cauchy(mu):
                ret = -0.1
                while ret <= 0.0:
                    ret = stats.cauchy.rvs(mu, 0.1)
                return ret

            pop_sf = np.fromiter((cauchy(mu) for mu in mu_sf), dtype=np.float64, count=self.population_size)

            new_pop = np.fromiter(itertools.chain.from_iterable(
                (self.currentToPBest1Bin(i, p_max, pop_cr[i], pop_sf[i]) for i in range(self.population_size))),
                dtype=np.float64, count=self.population_size * self.num_dim)
            new_pop.shape = (self.population_size, self.num_dim)

            new_pop_values = self.evaluate_population(new_pop)

            for i in range(self.population_size):
                if new_pop_values[i] <= population_values[i]:
                    self.population[i] = new_pop[i]
                    population_values[i] = new_pop_values[i]

            generation += 1
        return self.best

    def evaluate_population(self, population):
        iter = (self.f(self.scale(args)) for args in population)
        self.nevals += len(population)
        ret = np.fromiter(iter, dtype=np.float64, count=len(population))
        best_index = np.argmin(ret)
        if ret[best_index] < self.best[0]:
            self.best = ret[best_index], self.scale(population[best_index])
        return ret

    def scale(self, args):
        iter = (args[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0] for i in range(self.num_dim))
        return np.fromiter(iter, dtype=np.float64, count=self.num_dim)

    def currentToPBest1Bin(self, current, p_max, cross_rate, scaling_factor):
        # Todo from the actual best
        pbesti = np.random.randint(0, p_max)
        r1 = current
        r2 = current
        while r1 == current:
            r1 = np.random.randint(0, self.population_size)
        while r2 == current or r2 == r1:
            r2 = np.random.randint(0, self.population_size)
        random_variable = np.random.randint(0, self.num_dim)
        ret = np.empty(self.num_dim)
        for i in range(self.num_dim):
            if np.random.random() < cross_rate or i==random_variable:
                ret[i] = self.population[current][i] + scaling_factor * \
                            (self.population[pbesti][i] - self.population[current][i]) + scaling_factor * \
                            (self.population[r1][i] - self.population[r2][i]);
            else:
                ret[i] = self.population[current][i];

            # Fixup bounds
            if ret[i] < 0.0:
                ret[i] = self.population[current][i] * 0.5
            elif ret[i] > 1.0:
                ret[i] = (1.0 - self.population[current][i]) * 0.5
        return ret
