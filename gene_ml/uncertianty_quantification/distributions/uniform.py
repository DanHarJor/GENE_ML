import numpy as np

class Uniform():
    def __init__(self, bounds, seed=42):
        self.bounds = bounds
        self.rng = np.random.default_rng(seed=seed)
        self.uniform = lambda n_samples: self.rng.uniform(*bounds, n_samples)
    def sample(self, n_samples):
        # should return an array of samples from the distribution.
        return self.uniform(n_samples)