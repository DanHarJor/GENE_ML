# sampler/uniform.py
import numpy as np
class Uniform():
    """
    DOCSTRING
    """
    def __init__(self, bounds, num_samples, parameters):
        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = num_samples
        self.samples = self.generate_samples()
        self.current_index = 0

    def generate_samples(self):
        num_gen = np.random.default_rng(seed=38756)
        samples = {}
        for param, bound in zip(self.parameters,self.bounds):
            samples[param] = num_gen.uniform(*bound, self.num_samples)

        self.samples = samples

        return samples
    
    def get_next_sample(self):
        if self.current_index < len(self.samples):
            sample_dict = {key: value[self.current_index] for (key, value) in self.sample.items()}
            self.current_index += 1
            return sample_dict
        else:
            return None  # TODO: implement when done iterating!
