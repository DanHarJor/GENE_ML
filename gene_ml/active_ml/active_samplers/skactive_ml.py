from skactiveml.pool import GreedySamplingX, GreedySamplingTarget
from sklearn.ensemble import BaggingRegressor
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor
from skactiveml.utils import call_func, is_labeled
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import trapz

class skActiveML():
    def __init__(self, function, regressor, random_state=42):
        self.function = function
        self.regressor = regressor # must have a fit and predict function
        self.query_sampler = GreedySamplingX(random_state=random_state)
        self.integrals = []
        self.rmse_s = []
        self.integrals_dif = None
    def generate_samples(self, num_samples, pool_X, initial_X=None, initial_y=None, test_mode=False, integration_interval=None, plot_interval=None):
        if type(initial_X) != type(None) and type(initial_y) != type(None):
            X_all = np.concatenate((initial_X, pool_X), axis = 0)
            y_all = np.concatenate((initial_y, np.repeat(np.nan, len(pool_X))), axis = 0)
        else:
            X_all = pool_X
            y_all = np.repeat(np.nan, len(pool_X))
        if np.isscalar(X_all[0]):
            X_all = X_all.reshape(-1,1)
        
        if test_mode:
            X_test = np.linspace(X_all.min(), X_all.max(), 300).reshape(-1, 1)
            y_test = self.function(X_test)
            y_true = self.function(X_all)   
            integral_true = trapz(y_test.flatten(), X_test.flatten())         

        for i in range(num_samples):
            X = X_all[~np.isnan(y_all)]
            y = y_all[~np.isnan(y_all)]
            self.regressor.fit(X, y)
            indices, utils = call_func(self.query_sampler.query,
                X=X_all,
                y=y_all,
                reg=self.regressor,
                ensemble=SklearnRegressor(BaggingRegressor(self.regressor, n_estimators=4)),
                fit_reg=True,
                return_utilities=True,
            )
            
            old_is_lbld = is_labeled(y_all)
            y_all[indices] = self.function(X_all[indices])
            is_lbld = is_labeled(y_all)

            if test_mode:
                y_pred = self.regressor.predict(X_test)
                self.rmse_s.append(np.sqrt(np.mean((y_pred.flatten()-y_test.flatten())**2)))
        
                if integration_interval != None and i%integration_interval==0:
                    self.integrals.append(trapz(y_pred.flatten(), X_test.flatten()))

                if plot_interval!=None and i%plot_interval==0:
                    # print(integral_true, self.integrals[-1])
                    # print('debug, i, int dif', i, self.integrals[-1]-integral_true)
                    _, utilities_test = call_func(self.query_sampler.query,
                        X=X_all,
                        y=y_all,
                        reg=self.regressor,
                        ensemble=SklearnRegressor(BaggingRegressor(self.regressor, n_estimators=4)),
                        candidates=X_test,
                        fit_reg=True,
                        return_utilities=True,
                    )
                
                    figure, ax = plt.subplots(1,1)
                    ax_t = ax.twinx()
                    ax_t.plot(X_test, utilities_test.flatten(), c='green')
                    ax.scatter(X_all[~is_lbld], y_true[~is_lbld], c='lightblue')
                    ax.scatter(X_all[old_is_lbld], y_all[old_is_lbld], c='orange')
                    ax.scatter(X_all[indices], y_all[indices], c='red')
                    ax.plot(X_test, y_pred, c='black')
                    ax.set_title(self.query_sampler.__class__.__name__, fontdict={'fontsize': 15})

                    plt.show()
        self.integrals_dif =  np.abs(np.array(self.integrals)-integral_true)
        if test_mode and plot_interval != None:
            fig, ax = plt.subplots(1,1)
            iterations = np.arange(num_samples)
            ax.plot(iterations, self.rmse_s, label='RMSE', c='red')
            ax_t = ax.twinx()
            ax_t.plot(iterations, self.integrals_dif, label='Integral Difference (right axis)', c='green')
            fig.legend()
            fig.show()

    