from matplotlib import pyplot as plt
import numpy as np

def residual_plot(ax, fig, y_true, y_predicted, var_name, title=None, y_pred_err=None):
    mse=np.mean((y_true-y_predicted)**2)
    cb = ax.scatter(y_true, y_predicted, marker='X', c=y_pred_err, cmap='summer')
    if type(y_pred_err)!=type(None):
        fig.colorbar(cb, ax=ax)
    ref = np.linspace(0,np.max((np.max(y_true),np.max(y_predicted))), 100).astype(int)
    ax.plot(ref,ref, 'r')
    ax.set_xlabel(f'True {var_name}')
    ax.set_ylabel(f'Predicted {var_name}')
    if title != None:
        ax.set_title(title)
    ax.annotate(f'MSE: {np.round(mse,3)}',
            xy=(.65, .02), xycoords='axes fraction',fontsize=10)

def residual_hist(ax, fig, y_true, y_predicted, var_name, title=None):
    ax.hist(y_true-y_predicted, density=True, bins=20)
    ax.set_xlabel(f'Residuals, {var_name}')
    ax.set_ylabel(f'Normalised Frequency')
    if title != None:
        ax.set_title(title)
    