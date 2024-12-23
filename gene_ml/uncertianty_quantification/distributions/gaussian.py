import numpy as np

class Gaussian():
    def __init__(self, mean, sigma, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.gaussian = lambda n_samples: self.rng.normal(loc=mean, scale=sigma, size=n_samples)
    def sample(self, n_samples):
        # should return an array of samples from the distribution.
        return self.gaussian(n_samples)