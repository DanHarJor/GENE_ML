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
    def __init__(self, name, dim, kernel_type='matern32', fixed_kernel_args={"variance":1}):
        super().__init__(name)
        self.regressor = None
        self.model_type_id = 'gpr'
        self.kernel_type = kernel_type
        self.fixed_kernel_args = fixed_kernel_args
        self.dim = dim
        if dim ==1: dim=2
        if self.kernel_type == 'rational_quadratic':
            self.kernel = RatQuad(input_dim=dim, **fixed_kernel_args)
        elif self.kernel_type == 'matern32':
            self.kernel = Matern32(input_dim=dim)
        elif self.kernel_type == 'matern52':
            self.kernel = Matern52(input_dim=dim)
        elif self.kernel_type == 'radial_basis_function':
            self.kernel = RBF(input_dim=dim, **fixed_kernel_args)
        else: raise ValueError('kernel_type must be a valid kernel.')


    def train(self, x, y):
        print('GPR is a parameterless approach and does not have a training step. Instead the hyperparameters are tuned to the data.')
    
    def tune_hypers(self, x, y):
        if self.dim == 1:
            x = np.array(x)[:, None]
        y = np.array(y)[:, None] # GPy needs a 2D array
        print('OPTIMISING THE HYPERPERS')
        self.regressor = GPRegression(x, y, self.kernel, noise_var=0)
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
    
    def update_data(self, x,y):
        self.regressor = GPRegression(x,y, self.kernel)
    
    def fit(self, *args, **kargs):
        if not type(self.fixed_kernel_args)==type(None):
            self.train(*args, **kargs)
            self.tune_hypers(*args, **kargs)
        else:
            self.update_data(*args, **kargs)
    
if __name__ == '__main__':
    print('hello world')