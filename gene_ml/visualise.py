from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np

def residual_plot(ax, fig, y_true, y_predicted, var_name, title=None, y_pred_err=None):
    mse=np.mean((y_true-y_predicted)**2)
    if type(y_pred_err)!=type(None):
        ax.errorbar(y_true, y_predicted, yerr=y_pred_err, fmt='.', ecolor='red')
        cb = ax.scatter(y_true, y_predicted, marker='.', c=y_pred_err, cmap='summer', zorder=10)
        fig.colorbar(cb, ax=ax)
        
    else:
        print('NO UNCERTAINTY PROVIDED')
        ax.scatter(y_true, y_predicted, marker='.')
     

    print('MAX',np.max(y_true),np.max(y_predicted))
    ref = np.linspace(np.min((np.min(y_true),np.min(y_predicted))),np.max((np.max(y_true),np.max(y_predicted))), 100).astype(float)
    print(np.max(ref))
    ax.plot(ref,ref, 'r')
    ax.set_xlabel(f'Parent Model {var_name}')
    ax.set_ylabel(f'Surrogate Model {var_name}')
    if title != None:
        ax.set_title(f'{title}: {len(y_true)} data points')
    ax.annotate(f'MSE: {Decimal(mse):.2E}',
            xy=(.6, .02), xycoords='axes fraction',fontsize=10)

def residual_hist(ax, fig, y_true, y_predicted, var_name, title=None, bins=50):
    ax.hist(y_true-y_predicted, density=True, bins=bins)
    ax.set_xlabel(f'Residuals, {var_name}')
    ax.set_ylabel(f'Normalised Frequency')
    if title != None:
        ax.set_title(title)
    