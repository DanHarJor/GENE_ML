import os
import pickle
from scipy import stats
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

class Results():
    def __init__(self, name, *args, **kargs):
        self.name = name

        if not os.path.exists('saved_results'):
            os.mkdir('saved_results')
        self.path = os.path.join('saved_results',name)
    
    def save(self):
        pickle.dump(self, open(self.path, 'wb'))

    def load(self):
        return pickle.load(open(self.path, 'rb'))
    
    def exists(self):
        return os.path.exists(self.path)
    
class ResultsTraining(Results):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.growthrate_predict_seen = None
        self.growthrate_predict_unseen = None
        self.growthrate_predict_seen_errors = None
        self.growthrate_predict_unseen_errors = None

class ResultsGroundTruthTest(Results):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.altered_parameters_names = []
        self.growthrates = []
        self.frequencies = []








from scipy.special import kl_div as kl
from scipy.special import rel_entr as re
def kl_div(result_uq1, result_uq2):
    # print('PDFS',result_uq1.pdf, result_uq2.pdf)
    growthrates1 = result_uq1.growthrate_predict_uqsamples#only_first_return(model.predict(samples))
    KL = []
    res = np.arange(3,200,5)
    for r in res:
        p1, x = result_uq1.create_pdf(r)
        p2, x = result_uq2.create_pdf(r)
        KL.append(np.sum(p1 * np.log(p1/p2)))
    figure1 = plt.figure()
    plt.plot(res,KL)
    plt.xlabel('Number of PDE discretisations')
    plt.ylabel('KL-divergence')
    res_chosen = 50
    p1, x = result_uq1.create_pdf(res_chosen)
    p2, x = result_uq2.create_pdf(res_chosen)
    figure = plt.figure()
    plt.title(f'with {res_chosen} discretisations of pde')
    plt.plot(x, p1, '-r', label = result_uq1.name)
    plt.plot(x, p2, '-g', label = result_uq2.name)
    plt.legend()
    plt.xlabel('grothrate')
    plt.ylabel('pde')
    return np.sum(p1 * np.log(p1/p2)), np.sum(kl(p1, p2)), np.sum(re(p1, p2))

class ResultsUQ(Results):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.growthrate_nominal = None
        self.growthrate_lower_bound = None
        self.growthrate_upper_bound = None
        self.growthrate_predict_uqsamples = None
        self.bounds = None
        self.nominal_parameters = None

        self.kde = None
        self.pdf = None
        self.cdf = None
        self.conf95_lower = None
        self.conf95_upper = None
    def create_pdf(self, res):
        # x = np.linspace(self.conf95_lower, self.conf95_upper, res)
        x = np.linspace(0.06, 0.16, res)
        pdf = self.kde(x)
        return pdf, x

    def vis_forward_uq(self, ax, nbins, model_name, xlim, title=None, do95conf=True):
        growthrates = self.growthrate_predict_uqsamples#only_first_return(model.predict(samples))
        max_growthrate = np.max(growthrates)
        min_growthrate = np.min(growthrates)
        nominal_growthrate = self.growthrate_nominal#only_first_return(model.predict(nominal_parameters))
        # lower_bound_gr = self.growthrate_lower_bound#only_first_return(model.predict(np.array([b[0] for b in bounds]).reshape(2,1).T))
        # upper_bound_gr = self.growthrate_upper_bound#only_first_return(model.predict(np.array([b[1] for b in bounds]).reshape(2,1).T))
        # nominal_parameters = self.nominal_parameters
        # print('GROWTHRATES',growthrates.shape,growthrates)
        self.kde = stats.gaussian_kde(growthrates)
        print('GR MAX MIN',min_growthrate, max_growthrate)
        hist_x = np.linspace(min_growthrate,max_growthrate, 1000)
        n, bins, _ = ax.hist(growthrates, bins=nbins, density=True)
        if type(self.pdf)==type(None):
            self.pdf = self.kde(hist_x)
        print(do95conf)
        if do95conf:
            if type(self.cdf)==type(None):
                print('COMPUTING CDF')
                dx = (max_growthrate-min_growthrate)/1000
                ar = np.array([self.kde(xi) for xi in np.arange(min_growthrate, max_growthrate, dx)]).flatten()*dx
                self.cdf = np.cumsum(ar)
                self.save()
            self.conf95_lower = hist_x[np.argmin(abs(self.cdf-0.025))]
            self.conf95_upper = hist_x[np.argmin(abs(self.cdf-0.975))]
            ax.vlines(self.conf95_lower, 0, max(n), 'grey', label="95% confidance")
            ax.vlines(self.conf95_upper, 0, max(n), 'grey')
        
        # print('CDF',self.cdf)
        # print('SUM', sum(self.pdf), self.cdf[-1])
        # print('ARGMIN', np.argmin(abs(self.cdf-0.225)), np.argmin(abs(self.cdf-0.975)))
        print('CONF 95', self.conf95_lower, self.conf95_upper)
        ax.plot(hist_x, self.pdf)#, label='Gaussian Kernel Density Estimate')
        ax.set_xlabel('Growthrate')
        ax.set_ylabel('Probability Density')
        ax.vlines(nominal_growthrate, 0, max(n), 'r', label='Nominal Value')
        mean = np.mean(growthrates)
        var = np.var(growthrates)
        print('MEAN VAR',mean, var)
        ax.vlines(mean, 0, max(n), 'black', label='Mean')
        # ax.vlines(lower_bound_gr, 0, max(n), 'g', label='Lower Bound')
        # ax.vlines(upper_bound_gr, 0, max(n), 'm', label='Upper Bound')
        # ax.annotate(fr'{model_name}| $\nabla T_e$: {nominal_parameters[0][0]}$\pm$1.35  -  $\nabla T_i$: {nominal_parameters[0][1]}$\pm$0.7  -  Gaussian Error 95% Confidance',
                # xy=(0, 1.05), xycoords='axes fraction',fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(title)
        ax.legend()



if __name__ == '__main__':
    ruq = ResultsUQ(name='test_name')

    ruq.growthrate_predict_seen = 5
    ruq.nominal_growthrate = 3

    print(ruq.growthrate_predict_seen, ruq.nominal_growthrate)
