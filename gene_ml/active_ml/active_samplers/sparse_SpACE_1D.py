import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

import sparseSpACE
import numpy as np
from sparseSpACE.spatiallyAdaptiveCell import *
from sparseSpACE.Function import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *


class SparseSpACE_1D():
    def __init__(self, function, regressor, random_state=42):
        self.function = function
        self.regressor = regressor # must have a fit, predict function. predict must return uncertainty
        self.integrals = []
        self.rmse_s = []
        self.integrals_dif = None    

    def generate_samples(self, num_samples, X_range, initial_X=None, initial_y=None, test_mode=False, integration_interval=None, plot_interval=None):
        # Sparse space function needs to have eval def
        # function
        from sparseSpACE.Function import Function
        import numpy as np

        class f_wrap(Function):
            def __init__(self, a, b, function):
                super().__init__()
                self.a = a
                self.b = b
                self.function = function

            def eval(self,X):
                if any(np.array(X) >= self.b) or any(np.array(X) <= self.a):
                    return 0
                else:
                    return self.function(X)#np.sin(X)
            
            # def __call__(self,X):
            #     return np.sin(X)
        
        #dimension of the problem
        dim = 1
        # define integration domain boundaries
        a = [X_range[0]]
        b = [X_range[1]]

        f = f_wrap(a[0], b[0], self.function)
        # f = mysine(a=a[0],b=b[0])#GenzDiscontinious(border=midpoint,coeffs=coefficients)

        # define error estimator for refinement
        errorOperator=ErrorCalculatorSurplusCell()
        grid=TrapezoidalGrid(a=a, b=b)
        operation = Integration(f=f, grid=grid, dim=dim)#, reference_solution = reference_solution)

        
        if test_mode:
            X_test = np.linspace(*X_range, 300).reshape(-1,1)
            y_test = self.function(X_test).reshape(-1,1) #caution function might be heavy
            integral_true = trapz(y_test.flatten(), X_test.flatten()) 

        for i in range(3,num_samples+1,2): #must be odd
            # define Cell refinement strategy for Spatially Adaptive Combination Technique
            # define equidistant grid
            adaptiveCombiInstanceCell = SpatiallyAdaptiveCellScheme(a, b, operation=operation)
            adaptiveCombiInstanceCell.performSpatiallyAdaptiv(lmin=2, lmax=2, errorOperator=errorOperator, max_evaluations=i, do_plot=False, do_last_plot=False, print_output=False)
            
            points = adaptiveCombiInstanceCell.get_points()
            px = points
            y = self.function(px) #must be changed to get values from sparsespace for heavy functions
            self.regressor.fit(px, y)

            if test_mode:
                y_pred_test, uncertainty = self.regressor.predict(X_test, return_std=True)
                self.rmse_s.append(np.sqrt(np.mean((y_pred_test.flatten()-y_test.flatten())**2)))

            if integration_interval != None and i%integration_interval==0:
                self.integrals.append(trapz(y_pred_test.flatten(), X_test.flatten()))

            if plot_interval!=None and i%plot_interval==0:
                figure, ax = plt.subplots(1,1)
                ax.plot(X_test, y_pred_test, c='green')
                ax.fill_between(X_test.flatten(), y_pred_test-uncertainty, y_pred_test+uncertainty, color = [0.1,1,0.1,0.2])
                ax.scatter(X_test, y_test, c='lightblue')
                ax.scatter(px, y, c='orange')
                # ax.scatter(X_new, y_new, c='red')
                # ax.plot(X_test, y_pred_test, c='black')
                ax.set_title(f'Number of Points: {i}')
                plt.show()

        self.integrals_dif = np.abs(np.array(self.integrals)-integral_true)
        if test_mode and plot_interval != None:
            fig, ax = plt.subplots(1,1)
            iterations = np.arange(3,num_samples+1, 2)
            ax.plot(iterations, self.rmse_s, label='RMSE', c='red')
            ax_t = ax.twinx()
            ax_t.plot(iterations, self.integrals_dif, label='Integral Difference (right axis)', c='green')
            fig.legend()
            fig.show()   



        