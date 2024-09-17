%matplotlib inline
import numpy as np

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *

class sparseSpACE():
    def __init__(parameters, bounds, ):
        distributions = [("Uniform", b[0], b[1]) for b in bounds]

    