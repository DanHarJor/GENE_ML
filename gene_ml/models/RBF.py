try:
    from .base import Model
except:
    try:
        from base import Model
    except:
        raise ImportError 

from scipy.interpolate import RBFInterpolator

class RBF(Model):
    def __init__(self, name):
        super().__init__(name)
        self.regressor = None
        
    def train(self, x, y):
        self.regressor = RBFInterpolator(x,y)
    
    def tune_hypers(self, x, y):
        print('RBF INTERPOLATION HAS NO HYPERPARAMETERS')

    def predict(self, x):
        y_predict = self.regressor(x)
        return y_predict
    
    
    
if __name__ == '__main__':
    print('hello world')