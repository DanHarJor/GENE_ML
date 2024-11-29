try:
    from .base import Model
except:
    try:
        from base import Model
    except:
        raise ImportError 

from GPy.models import GPRegression
from GPy.kern import RBF
from GPy.kern import Matern32
from GPy.kern import Matern52
from GPy.kern import RatQuad
import numpy as np

class GPR(Model):
    def __init__(self, name, kernel_type='matern32'):
        super().__init__(name)
        self.regressor = None
        self.model_type_id = 'gpr'
        self.kernel_type = kernel_type

    def train(self, x, y):
        print('GPR is a parameterless approach and does not have a training step. Instead the hyperparameters are tuned to the data.')
    
    def tune_hypers(self, x, y):
        if self.kernel_type == 'rational_quadratic':
            kernel = RatQuad(input_dim=len(x[0]))
        elif self.kernel_type == 'matern32':
            kernel = Matern32(input_dim=len(x[0]))
        elif self.kernel_type == 'matern52':
            kernel = Matern52(input_dim=len(x[0]))
        elif self.kernel_type == 'radial_basis_function':
            kernel = RBF(input_dim=len(x[0]))
        else: raise ValueError('kernel_type must be a valid kernel.')

        x = np.array(x)
        y = np.array(y)[:, None] # GPy needs a 2D array
        print('OPTIMISING THE HYPERPERS')
        self.regressor = GPRegression(x, y, kernel, noise_var=1.0)
        print('CURRENT HYPERS:\n', self.regressor)
        print('OPTIMISING THE HYPERPERS:')
        self.regressor.optimize_restarts(num_restarts = 10)

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