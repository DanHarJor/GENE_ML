from base import Model
import sys
import os
pathap = os.path.join('GENE_ML','gene_ml','static_sparse_grid_approximations')
sys.path.append(pathap)

from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *
import numpy as np


class SSG_POLY(Model):
    def __init__(self, ssg_sampler):        
        self.InterpToSpectral_obj = InterpolationToSpectral(ssg_sampler.dim, ssg_sampler.level_to_nodes, ssg_sampler.left_bounds, ssg_sampler.right_bounds, ssg_sampler.weights, ssg_sampler.level, ssg_sampler.Grid_obj)

        self.multiindex_set = ssg_sampler.Multiindex_obj.get_std_total_degree_mindex(ssg_sampler.level)
    
    def train(self, x, y):
        for m, multiindex in enumerate(self.multiindex_set):
            for point, label in zip(x,y):
                print('X', point)
                print('Y', label)
                self.InterpToSpectral_obj.update_sg_evals_all_lut(point,label)
                print('MULTIINDEX',multiindex)
            # self.InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, ssg_sampler.Grid_obj)
    
    def predict(self, x):
        # should return a prediction and its errors if available
        f_interp = lambda x: self.InterpToSpectral_obj.eval_operation_sg(self.multiindex_set, x)
        return f_interp(x)

if __name__ == '__main__':
    f_ref = lambda x: x[0]**2 + x[1]**2 + x[2]**3
    n_train = 10
    dim = 3
    x = np.random.uniform(0,1,size=(3,n_train)).T
    y = f_ref(x)
    print(x, y)
    print(x.shape,y.shape)
    
    sys.path.append(os.path.join('GENE_ML','gene_ml'))
    from samplers.static_sparse_grid import StaticSparseGrid
    parameters = ['love', 'peace', 'harmony']
    bounds = [(1,2),(300,400),(5000,6000)]
    ssg_sampler = StaticSparseGrid(bounds, parameters, level=3)
    poly = SSG_POLY(ssg_sampler)
    poly.train(x,y)
    
    n_test = 50
    test = np.random.uniform(0,1,size=(3,1000))
    predicted = poly.predict(test)
    true = f_ref(test) 
    print(predicted)
    print(true)


        
    
