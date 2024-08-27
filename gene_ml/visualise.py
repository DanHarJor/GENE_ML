from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np

from scipy import stats

def residual_plot(ax, fig, y_true, y_predicted, var_name, title=None, y_pred_err=None, font_size=None):
    residuals = y_true - y_predicted
    mse=np.mean((y_true-y_predicted)**2)
    if type(y_pred_err)!=type(None):
        ax.errorbar(y_true, residuals, yerr=y_pred_err, fmt='.', ecolor='red')
        # cb = ax.scatter(y_true, residuals, marker='.', c=y_pred_err, cmap='summer', zorder=10)
        # fig.colorbar(cb, ax=ax)
        
    else:
        print('NO UNCERTAINTY PROVIDED')
        ax.scatter(y_true, residuals, marker='.')
     

    refx = np.linspace(np.min(y_predicted),np.max(y_predicted),100)#np.linspace(np.min((np.min(y_true),np.min(y_predicted))),np.max((np.max(y_true),np.max(y_predicted))), 100).astype(float)
    refy = np.zeros(len(refx))
    ax.plot(refx,refy, 'r')
    ax.set_xlabel(f'GENE {var_name} (ground truth)', fontsize=font_size)
    ax.set_ylabel(f'Residuals, {var_name}', fontsize=font_size)
    if title != None:
        ax.set_title(f'{title}', fontsize=font_size)##: {len(y_true)} data points')
    ax.annotate(f'MSE: {Decimal(mse):.2E}',
            xy=(.7, .02), xycoords='axes fraction', fontsize=font_size)
    
    ax.set_xlim(left=0)

def residual_hist(ax, fig, y_true, y_predicted, var_name, title=None, bins=50, orientation='horizontal'):
    ax.hist(y_true-y_predicted, density=True, bins=bins, orientation=orientation)
    if orientation == 'horizontal':
        # ax.set_ylabel(f'Residuals, {var_name}')
        ax.set_xlabel(f'Normalised Frequency')
    if orientation == 'verticle':
        ax.set_xlabel(f'Residuals, {var_name}')
        ax.set_ylabel(f'Normalised Frequency')

    if title != None:
        ax.set_title(title)

def forward_uq(ax, results_uq, nbins, model_name, xlim):
            
    growthrates = results_uq.growthrate_predict_uqsamples#only_first_return(model.predict(samples))
    nominal_growthrate = results_uq.growthrate_nominal#only_first_return(model.predict(nominal_parameters))
    lower_bound_gr = results_uq.growthrate_lower_bound#only_first_return(model.predict(np.array([b[0] for b in bounds]).reshape(2,1).T))
    upper_bound_gr = results_uq.growthrate_upper_bound#only_first_return(model.predict(np.array([b[1] for b in bounds]).reshape(2,1).T))
    nominal_parameters = results_uq.nominal_parameters
    # print('GROWTHRATES',growthrates.shape,growthrates)
    kde = stats.gaussian_kde(growthrates)
    hist_x = np.linspace(np.min(growthrates),0.08, 1000)
    n, bins, _ = ax.hist(growthrates, bins=nbins, density=True)
    ax.plot(hist_x, kde(hist_x), label='Gaussian Kernel Density Estimate')
    ax.set_xlabel('Growthrate')
    ax.set_ylabel('Probability Density')
    ax.vlines(nominal_growthrate, 0, max(n), 'r', label='Nominal Value')
    mean = np.mean(growthrates)
    var = np.var(growthrates)
    print('MEAN VAR',mean, var)
    ax.vlines(mean, 0, max(n), 'black', label='Mean')
    ax.vlines(mean+np.sqrt(var), 0, max(n), 'grey', label='Standard Deviation')
    ax.vlines(mean-np.sqrt(var), 0, max(n), 'grey')
    ax.vlines(lower_bound_gr, 0, max(n), 'g', label='Lower Bound')
    ax.vlines(upper_bound_gr, 0, max(n), 'm', label='Upper Bound')
    ax.annotate(fr'{model_name}| $\nabla T_e$: {nominal_parameters[0][0]}$\pm$1.35  -  $\nabla T_i$: {nominal_parameters[0][1]}$\pm$0.7  -  Gaussian Error 95% Confidance',
            xy=(0, 1.05), xycoords='axes fraction',fontsize=10)
    ax.set_xlim(xlim)
    ax.legend()
