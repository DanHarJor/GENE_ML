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
        self.model_type_id = 'gpr'

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
    
    def predict(self, x, disclude_errors=False):
        input = np.array(x)
        if len(input.shape) == 1: #prediction for one point
            input = np.array([x])
        y_predict, y_var = self.regressor.predict(input)
        # print(y_predict.shape)
        # print(y_predict) 
        if disclude_errors:
            return y_predict[:,0] 
        else:
            y_2sig = np.sqrt(y_var[:,0]) * 2 
            return [y_predict[:,0], y_2sig]
    
    
    def fit(self, *args, **kargs):
        self.train(*args, **kargs)
        self.tune_hypers(*args, **kargs)
    
if __name__ == '__main__':
    print('hello world')