# sampler/uniform.py
import numpy as np
class Gaussian():
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
        num_gen = np.random.default_rng(seed=38756)
        samples = {}
        for param, bound in zip(self.parameters,self.bounds):
            if type(bound) == type((0.1,0.1)):
                conf_95 = (bound[1]-bound[0])*0.5 #length from the mean to 95%confidance interval. Assuming bounds given are the limits of the 95%confidance interval
                mean = bound[0]+(bound[1]-bound[0])*0.5 # if gaussian mean is center of bounds. 
                samples[param] = num_gen.normal(loc=mean, scale=conf_95/2, size=self.num_samples)
            elif type(bound) == type(0.1): #if bound is a float then it is static and not scanned.
                samples[param] = np.repeat(bound, self.num_samples)
        self.samples = samples
        return samples
            
    def get_next_sample(self):
        if self.current_index < len(self.samples):
            sample_dict = {key: value[self.current_index] for (key, value) in self.sample.items()}
            self.current_index += 1
            return sample_dict
        else:
            return None  # TODO: implement when done iterating!
