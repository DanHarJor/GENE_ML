try:
    from .base import Model
except:
    try:
        from base import Model
    except:
        raise ImportError 
from scipy.interpolate import interp1d
import numpy as np

class Interp_1D(Model):
    def __init__(self, name, kind='linear'):
        super().__init__(name)
        self.kind = kind
        self.interpolation = None

    def train(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        self.interpolation = interp1d(x, y, kind=self.kind)
    
    def tune_hypers(self, x, y):
        print('Interpolation has no hyper parameters apart from kind')
    
    def predict(self, x, return_std=False):
        x = x.flatten()
        pred = self.interpolation(x)
        if return_std:
            return pred, np.zeros(len(pred))
        else:
            return pred
    
    def fit(self, *args, **kargs):
        self.train(*args, **kargs)
        