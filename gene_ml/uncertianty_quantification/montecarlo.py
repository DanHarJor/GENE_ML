import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
        
class Montecarlo():
    def __init__(self, function, distributions, dimension_labels=None):
        self.function = function #must return a scalar, this is what we want to quantify the uncertianty of
        self.distributions = distributions #an array of distributions, one for each dimension of the problem, must be in the order that the function expects the input to be in  
        if dimension_labels == None:
            dimension_labels = np.arange(len(distributions)).astype('str')

        self.output_kde = None
        self.f = None
    def compute_output_distribution(self, n_samples):
        dist_samples = []
        for dist in self.distributions:
            print(dist.sample(n_samples).shape)
            dist_samples.append(dist.sample(n_samples))

        points = np.stack(dist_samples).T

        f = self.function(points)
        self.f = f
        kde = stats.gaussian_kde(f)
        self.output_kde = kde
        
    def plot_output_distribution(self, bins=100, nominal_parameters=None):
        grid_size = 1000
        plot_bounds = [(0,1),(0,1)]
        xlow, xhigh = plot_bounds[0][0], plot_bounds[0][1]
        ylow, yhigh = plot_bounds[1][0], plot_bounds[1][1]
        x = np.linspace(xlow, xhigh, grid_size)
        y = np.linspace(ylow, yhigh, grid_size)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        Zmax = self.function(pos)
        # Zdist = np.mean([dist(pos) for dist in self.distributions])

        out_dist = Zmax.flatten()
        kde_out = stats.gaussian_kde(out_dist)


        if type(self.output_kde) == type(None) or type(self.f) == type(None): 
            raise ValueError('Daniel Says, you must run compute output distribution before plotting')
        fig = plt.figure(dpi=200)
        hist_x = np.linspace(np.min(self.f),np.max(self.f), 1000)
        n, bins, _ = plt.hist(self.f, bins=bins, density=True)
        plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate')
        plt.plot(hist_x, kde_out(hist_x), color='red', label='trial')
        
        plt.xlabel('Function Value')
        plt.ylabel('Probability Density')
        # plt.vlines(nominal_growthrate, 0, max(n), 'r', label='Nominal Value')
        # plt.vlines(left_bound_gr, 0, max(n), 'g', label='Lower Uncertainty')
        # plt.vlines(right_bound_gr, 0, max(n), 'm', label='Upper Uncertainty')
        # # plt.annotate(fr'$\nabla T_e$: {nominal_parameters[0][0]}$\pm$1.35  -  $\nabla T_i$: {nominal_parameters[0][1]}$\pm$0.7  -  Gaussian Error 95% Confidance',
                    # xy=(0, 1.01), xycoords='axes fraction',fontsize=10)
        plt.legend()
        plt.show()


        

            

        
    