import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

import sparseSpACE
import numpy as np
from sparseSpACE.spatiallyAdaptiveCell import ErrorCalculatorSurplusCell, SpatiallyAdaptiveCellScheme
from sparseSpACE.Function import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import TrapezoidalGrid, Integration

from sparseSpACE.Function import Function
import numpy as np
class fun_wrap(Function):
    def __init__(self, function, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.function = function

    def eval(self,X):
        if any(np.array(X) >= self.b) or any(np.array(X) <= self.a):
            return 0
        else:
            return self.function(X)

class sparseSpACE_spatial():
    # This class covers the spatialy adaptive procedures found in the SparseSpACE tutorial.ipynb
    # https://github.com/obersteiner/sparseSpACE/blob/master/ipynb/Tutorial.ipynb
    # They are designed to get an accurate integral and not accurate sobel indicies or an accurate surrogate model
    def __init__(self, dim, function, regressor, scheme='cell'):
        self.function = fun_wrap(function)
        self.regressor = regressor # must have a fit and predict function
        
        # Performance metrics. 
        self.integrals = [] #Designed to perform best here
        self.rmse_s = []
        self.sobel_indicies = []
        self.integrals_dif = None

        # Bounds, assumed to be 0 and 1
        a = np.zeros(dim)
        b = np.ones(dim)
        self.grid = TrapezoidalGrid(a,b)

        self.operation = Integration(self.function, self.grid, self.dim)

        if scheme == 'cell':
            self.errorOperator = ErrorCalculatorSurplusCell()
            self.adaptiveCombiInstance = SpatiallyAdaptiveCellScheme(a,b,self.operation)
    

    def generate_samples(self, test_mode=False):

    