import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from GENE_ML.gene_ml.uncertianty_quantification.distributions.uniform import Uniform


class Montecarlo():
    def __init__(self, function):
        self.function = function #must return a scalar, this is what we want to quantify the uncertianty of
        self.output_kde = None
        self.f = None

    def compute_output_distribution(self, n_samples, input_distributions):
        dist_samples = []
        for dist in input_distributions:
            print(dist.sample(n_samples).shape)
            dist_samples.append(dist.sample(n_samples))

        points = np.stack(dist_samples).T

        f = self.function(points)
        self.f = f
        kde = stats.gaussian_kde(f)
        self.output_kde = kde
        
    def plot_output_distribution(self, bins=100, nominal_parameters=None):
        # grid_size = 1000
        # plot_bounds = [(0,1),(0,1)]
        # xlow, xhigh = plot_bounds[0][0], plot_bounds[0][1]
        # ylow, yhigh = plot_bounds[1][0], plot_bounds[1][1]
        # x = np.linspace(xlow, xhigh, grid_size)
        # y = np.linspace(ylow, yhigh, grid_size)
        # X, Y = np.meshgrid(x, y)
        # pos = np.dstack((X, Y))
        # Zmax = self.function(pos)
        # # Zdist = np.mean([dist(pos) for dist in self.distributions])

        # out_dist = Zmax.flatten()
        # kde_out = stats.gaussian_kde(out_dist)


        if type(self.output_kde) == type(None) or type(self.f) == type(None): 
            raise ValueError('Daniel Says, you must run compute output distribution before plotting')
        fig = plt.figure(dpi=200)
        hist_x = np.linspace(np.min(self.f),np.max(self.f), 1000)
        n, bins, _ = plt.hist(self.f, bins=bins, density=True)
        plt.plot(hist_x, self.output_kde(hist_x), label='Gaussian Kernel Density Estimate')
        # plt.plot(hist_x, kde_out(hist_x), color='red', label='trial')
        
        plt.xlabel('Function Value')
        plt.ylabel('Probability Density')
        # plt.vlines(nominal_growthrate, 0, max(n), 'r', label='Nominal Value')
        # plt.vlines(left_bound_gr, 0, max(n), 'g', label='Lower Uncertainty')
        # plt.vlines(right_bound_gr, 0, max(n), 'm', label='Upper Uncertainty')
        # # plt.annotate(fr'$\nabla T_e$: {nominal_parameters[0][0]}$\pm$1.35  -  $\nabla T_i$: {nominal_parameters[0][1]}$\pm$0.7  -  Gaussian Error 95% Confidance',
                    # xy=(0, 1.01), xycoords='axes fraction',fontsize=10)
        plt.legend()
        plt.show()

    def uniform_uq(self, bounds, n_samples, n_bins):
        #This assums each dimensions is centered around a nominal value and the bounds are uncertanties.
        uniforms = []
        for bound in bounds:
            uniforms.append(Uniform(bound))
        self.compute_output_distribution(n_samples, input_distributions=uniforms)
        self.plot_output_distribution(n_bins)

    
import numpy as np
from scipy.stats import entropy

def relative_entropy(samples_p, samples_q, num_bins):
    """
    Calculate the relative entropy (Kullback-Leibler divergence) between two distributions.

    Parameters:
    samples_p (list or np.array): Samples from the first distribution.
    samples_q (list or np.array): Samples from the second distribution.

    Returns:
    float: The relative entropy between the two distributions.
    """
    # Convert samples to numpy arrays
    samples_p = np.array(samples_p)
    samples_q = np.array(samples_q)

    # Calculate the probability density functions
    p_values, _ = np.histogram(samples_p, bins=num_bins, density=True)
    q_values, _ = np.histogram(samples_q, bins=num_bins, density=True)

    # Add a small value to avoid division by zero and log of zero
    p_values += 1e-10
    q_values += 1e-10

    # Calculate the relative entropy
    rel_entropy = entropy(p_values, q_values)

    return rel_entropy

        

            

        
    