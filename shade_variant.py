import numpy as np

class Shade:
    def __init__(self, f, bounds, population_size, archive_size):
        self.bounds = np.array(list(bounds), ndmin=2)
        self.num_dim = len(self.bounds)
        self.f = f
        self.population_size = population_size
        self.archive_size = archive_size

    def optimize(self):
        population = np.random.random((self.population_size, self.num_dim))
        population_values = self.evaluate_population(population)
        best_index = np.argmin(population_values)
        self.best = population_values[best_index], self.scale(population[best_index])
        return self.best

    def evaluate_population(self, population):
        iter = (self.f(self.scale(args)) for args in population)
        return np.fromiter(iter, dtype=np.float64, count=len(population))

    def scale(self, args):
        iter = (args[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0] for i in range(self.num_dim))
        return np.fromiter(iter, dtype=np.float64, count=self.num_dim)
