# sampler/uniform.py
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale, discrepancy
from .sampler_base import Sampler
class LatinHyperCube(Sampler):
    """
    DOCSTRING
    """
    def __init__(self, bounds, num_samples, parameters):
        self.parameters = parameters
        self.bounds = bounds
        self.dim = len(parameters)
        self.num_samples = num_samples
        self.samples_array_norm = None
        self.samples_array = None
        self.samples = self.generate_samples()
        self.current_index = 0

        Sampler.__init__(self, self.samples_array_norm)

    def generate_samples(self, strength=1):
        lhc = LatinHypercube(d=self.dim, optimization="random-cd", strength=strength)
        self.samples_array_norm = lhc.random(n=self.num_samples)
        self.samples_array = scale(self.samples_array_norm, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        self.samples = {}
        for param, values in zip(self.parameters, self.samples_array.T):
            self.samples[param] = values        

        return self.samples
    
    def get_discrepancy(self):
        # discrepancy measures how uniformly spread the samples are
        return discrepancy(self.samples_array_norm)
    
    def get_next_sample(self):
        if self.current_index < len(self.samples):
            sample_dict = {key: value[self.current_index] for (key, value) in self.sample.items()}
            self.current_index += 1
            return sample_dict
        else:
            return None  # TODO: implement when done iterating!
        
if __name__ == "__main__":
    parameters = ['_grp_species_0-omt','_grp_species_1-omt','species-omn']
    bounds = [(4,6.7), (2.1,3.5), (0.16,2.9)] 
    sampler = LatinHyperCube(bounds, 9, parameters)
    print('discrepancy, 5 points', sampler.get_discrepancy())
    sampler2 = LatinHyperCube(bounds, 20, parameters)
    print('discrepancy, 20 points', sampler2.get_discrepancy())
    

