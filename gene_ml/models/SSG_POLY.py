try:
    from .base import Model
except:
    try:
        from base import Model
    except: ImportError
import sys
import os
import chaospy as cp
import time

# os.system("export PYTHONPATH=$PYTHONPATH:/home/djdaniel/GENE_UQ/GENE_ML/gene_ml/static_sparse_grid_approximations")

# pathap = os.path.join('home','djdaniel','GENE_UQ','GENE_ML','gene_ml','static_sparse_grid_approximations', 'sg_lib')
# sys.path.append(pathap)

from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *
from sg_lib.get_sparse_grid_model_closed_form import *
# from gene_ml.tools import sec_to_time_format
from GENE_ML.gene_ml.tools import sec_to_time_format
import numpy as np


class SSG_POLY(Model):
    def __init__(self, ssg_sampler, name):
        Model.__init__(self, name=name)        
        self.InterpToSpectral_obj = InterpolationToSpectral(ssg_sampler.dim, ssg_sampler.level_to_nodes, ssg_sampler.left_bounds, ssg_sampler.right_bounds, ssg_sampler.weights, ssg_sampler.level, ssg_sampler.Grid_obj)
        self.multiindex_set = ssg_sampler.Multiindex_obj.get_std_total_degree_mindex(ssg_sampler.level)
        
        self.ssg_sampler = ssg_sampler
        self.mean_est = None
        self.var_est = None
        self.all_sobol_indicies = None

        self.coeff_SG, self.basis_SG = None, None

        self.multiindex_bin = None
        
    def train(self, y):
        i = 0
        for m, multiindex in enumerate(self.multiindex_set):
            mindex_grid_inputs = self.ssg_sampler.Grid_obj.get_sg_surplus_points_multiindex(multiindex)
            for sg_point_sub in mindex_grid_inputs:
                if y[i] == -1: continue
                self.InterpToSpectral_obj.update_sg_evals_all_lut(sg_point_sub,y[i])#normalised point goes to unormalised label
                i += 1
            self.InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, self.ssg_sampler.Grid_obj)

        self.coeff_SG, self.basis_SG = self.InterpToSpectral_obj.get_spectral_coeff_sg(self.multiindex_set)

    
    def get_polynomial(self):
        if type(self.coeff_SG) != type(None) and type(self.basis_SG) != type(None):
            var_ND = cp.variable(self.ssg_sampler.dim)
            reduced_model_full = get_reduced_model_in_closed_form(self.coeff_SG, self.basis_SG, self.ssg_sampler.left_stoch_boundary, self.ssg_sampler.right_stoch_boundary, var_ND)
            # print(reduced_model_full)
            # print(type(reduced_model_full))
            return reduced_model_full
        else:
            raise EnvironmentError("You must train before you can get the polynomial. Please run SSG_POLY.train(labels)")


    def predict(self, x):
        start = time.time()
        polynomial = self.get_polynomial()
        prediction = polynomial(*x.T)
        # # should return a prediction and its errors if available
        # f_interp = lambda x: self.InterpToSpectral_obj.eval_operation_sg(self.multiindex_set, x)
        # #f_interp needs normalised points
        # normalise = lambda x: (x-self.ssg_sampler.left_stoch_boundary) / (self.ssg_sampler.right_stoch_boundary - self.ssg_sampler.left_stoch_boundary)
        # prediction = []
        # for xi in x:
        #     prediction.append(f_interp(normalise(xi)))
        end = time.time()
        print(f'PREDICTION WALL TIME FOR {len(x)} POINTS, dd-hh:mm:ss | {sec_to_time_format(end-start)}')
        return prediction

    def mean_var_est(self):
        self.mean_est = self.InterpToSpectral_obj.get_mean(self.coeff_SG)
        self.var_est = self.InterpToSpectral_obj.get_variance(self.coeff_SG)
        return self.mean_est, self.var_est
    
    def sobol_ind(self):
        if type(self.coeff_SG) == type(None): self.mean_var_est()

        multiindex_bin 		= self.ssg_sampler.Multiindex_obj.get_poly_mindex_binary(self.ssg_sampler.dim)
        self.all_sobol_indices 	= self.InterpToSpectral_obj.get_all_sobol_indices(multiindex_bin, self.coeff_SG, self.multiindex_set)
        return self.all_sobol_indices


if __name__ == '__main__':
    f_ref = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + 2*x[2] * x[3] + 3
    
    sys.path.append(os.path.join('GENE_ML','gene_ml'))
    from samplers.static_sparse_grid import StaticSparseGrid
    parameters = ['temp', 'gradn', 'gradTi','gradTe']
    bounds = [(0.2,1.3),(-0.4,2.4),(2.,4.),(1.,3.)]
    ssg_sampler = StaticSparseGrid(parameters=parameters, bounds=bounds, level=3, level_to_nodes=2)
    print('NUM SAMPLES',ssg_sampler.num_samples)

    x=ssg_sampler.samples_array
    y = f_ref(x.T)
    y[3] = -1

    poly = SSG_POLY(ssg_sampler)

    poly.train(y)

    left_bounds, right_bounds = [b[0] for b in bounds], [b[1] for b in bounds] 
    test_points = np.random.uniform(left_bounds, right_bounds, size=(10, 4))
    # test_points = np.random.uniform(0, 1, size=(10, 4))
    print('TEST LABELS AND TEST PREDICTIONS SHOULD BE THE SAME IF TRAINED ON A GROUND TRUTH POLYNOMIAL')
    print('test labels', f_ref(test_points.T))
    print('test predictions', poly.predict(test_points))

    print('MEAN & VAR ESTIMATE', poly.mean_var_est())

    print('SOBOL INDICIES', poly.sobol_ind())



        
    
