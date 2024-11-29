try:
    from .base import Model
except:
    try:
        from base import Model
    except:
        raise ImportError 

from scipy.interpolate import LinearNDInterpolator
import numpy as np
class LINEAR_INTERP(Model):
    def __init__(self, name):
        super().__init__(name)
        self.regressor = None
        
    def train(self, x, y):
        self.regressor = LinearNDInterpolator(x,y, rescale=True)
    
    def tune_hypers(self, x, y):
        print('LINEAR INTERPOLATION HAS NO HYPERPARAMETERS')

    def predict(self, x, return_std):
        y_predict = self.regressor(x)
        if return_std:
            return y_predict, np.zeros(len(y_predict))
        else:
            return y_predict
    
    
    
if __name__ == '__main__':
    print('hello world')