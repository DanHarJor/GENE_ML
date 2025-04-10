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

from scipy.spatial.distance import pdist
    

class GPR(Model):
    def __init__(self, name, dim, kernel_type='radial_basis_function', do_overfit=False, fixed_kernel_args=None):
        super().__init__(name)
        self.regressor = None
        self.model_type_id = 'gpr'
        self.kernel_type = kernel_type
        self.fixed_kernel_args = fixed_kernel_args
        self.dim = dim
        self.do_overfit = do_overfit
        if dim ==1: dim=2
        if self.kernel_type == 'rational_quadratic':
            self.kernel = RatQuad(input_dim=dim)#, **fixed_kernel_args)
            self.kernel_class = RatQuad
        elif self.kernel_type == 'matern32':
            self.kernel = Matern32(input_dim=dim)
            self.kernel_class = Matern32
        elif self.kernel_type == 'matern52':
            self.kernel = Matern52(input_dim=dim)
            self.kernel_class = Matern52
        elif self.kernel_type == 'radial_basis_function':
            self.kernel = RBF(input_dim=dim)#, **fixed_kernel_args)
            self.kernel_class = RBF
        else: raise ValueError('kernel_type must be a valid kernel.')


    def train(self, x, y):
        print('GPR is a parameterless approach and does not have a training step. Instead the hyperparameters are tuned to the data.')
    
    def tune_hypers(self, x, y):
        # GPy needs a 2d Array
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim == 1:
            x = x[:, None]

        print('OPTIMISING THE HYPERPERS')
        self.regressor = GPRegression(x, y, self.kernel, noise_var=0)
        print('CURRENT HYPERS:\n', self.regressor)
        print('OPTIMISING THE HYPERPERS:')
        self.regressor.Gaussian_noise.variance.fix(0)
        self.regressor.optimize_restarts(num_restarts = 10)

        print('RESULTING HYPERS:\n',self.regressor)

    
    def predict(self, x, disclude_errors=False):
        input = np.array(x)
        if input.ndim == 1:
            input = input[:, None]
        if len(input.shape) == 1 and len(input) == self.dim: #prediction for one point
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
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim == 1:
            x = x[:, None]
        print('DEBUG update', x.shape, y.shape)
        self.kernel = self.kernel_class(self.dim, **self.fixed_kernel_args)
        self.regressor = GPRegression(x, y, self.kernel, noise_var=0)
    
    def overfit(self,x,y):
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim == 1:
            x = x[:, None]
        amp = np.max(y)
        distances = pdist(x, 'euclidean')
        lengthscale = np.mean(distances)#np.percentile(distances, 25)
        self.kernel = self.kernel_class(self.dim, variance=amp, lengthscale=lengthscale)
        self.regressor = GPRegression(x, y, self.kernel, noise_var=0)
    
    def fit(self, *args, **kargs):
        if type(self.fixed_kernel_args)==type(None) and not self.overfit:
            self.train(*args, **kargs)
            self.tune_hypers(*args, **kargs)
        elif not type(self.fixed_kernel_args)==type(None):
            self.update_data(*args, **kargs)
        elif self.do_overfit:
            self.overfit(*args, **kargs)
        else:
            self.train(*args, **kargs)
            self.tune_hypers(*args, **kargs)

    
if __name__ == '__main__':
    print('hello world')