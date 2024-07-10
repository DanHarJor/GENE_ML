# sampler/uniform.py
import numpy as np
class Grid():
    """
    DOCSTRING
    """
    def __init__(self, bounds, num_samples, parameters):
        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = num_samples
        self.samples = self.generate_samples()
        self.samples_np = np.stack(list(self.samples.values())).T
        self.current_index = 0

    def generate_samples(self):
        samples = {}
        for param, bound in zip(self.parameters,self.bounds):
            samples[param] = np.linspace(*bound, self.num_samples)
        self.samples = samples
        return samples
    
    def get_next_sample(self):
        if self.current_index < len(self.samples):
            sample_dict = {key: value[self.current_index] for (key, value) in self.sample.items()}
            self.current_index += 1
            return sample_dict
        else:
            return None  # TODO: implement when done iterating!
        
if __name__ == "__main__":
    parameters = ['_grp_species_1-omt', '_grp_species_0-omt']
    bounds = [(4,6.7), (2.1,3.5)]
    sampler = Grid(bounds, 4, parameters)

    print(sampler.samples)
