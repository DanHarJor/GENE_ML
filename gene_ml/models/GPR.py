try:
    from .base import Model
except:
    try:
        from base import Model
    except:
        raise ImportError 

from GPy.models import GPRegression
from GPy.kern import RBF
import numpy as np

class GPR(Model):
    def __init__(self, name):
        super().__init__(name)
        self.regressor = None

    def train(self, x, y):
        print('GPR is a parameterless approach and does not have a training step. Instead the hyperparameters are tuned to the data.')
    
    def tune_hypers(self, x, y):
        x = np.array(x)
        y = np.array(y)[:, None] # GPy needs a 2D array
        print('OPTIMISING THE HYPERPERS')
        kernel = RBF(input_dim=len(x[0]), variance=1., lengthscale=10.,ARD=True)
        self.regressor = GPRegression(x, y, kernel, noise_var=1.0)
        print('CURRENT HYPERS:\n', self.regressor)
        print('OPTIMISING THE HYPERPERS:')
        self.regressor.optimize_restarts(num_restarts = 3)
        print('RESULTING HYPERS:\n',self.regressor)
    
    def predict(self, x):
        y_predict, y_error = self.regressor.predict(np.array(x))
        print(y_predict.shape)
        print(y_predict) 
        return [y_predict[:,0], y_error[:,0]]
    
    
    
if __name__ == '__main__':
    print('hello world')