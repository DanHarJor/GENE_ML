from matplotlib import pyplot as plt
import numpy as np

def residuals(y_true, y_predicted, var_name, title=None, y_pred_err=None):
    mse=np.mean((y_true-y_predicted)**2)

    
    
    
    ncol = 2
    size=6
    fig, [ax1, ax2] = plt.subplots(1,ncol, figsize=(size*ncol,size))
    
    cb = ax1.scatter(y_true, y_predicted, marker='X', c=y_pred_err, cmap='summer')
    if type(y_pred_err)!=type(None):
        fig.colorbar(cb, ax=ax1)
    ref = np.linspace(0,np.max((np.max(y_true),np.max(y_predicted))), 100).astype(int)
    ax1.plot(ref,ref, 'r')
    ax1.set_xlabel(f'True {var_name}')
    ax1.set_ylabel(f'Predicted {var_name}')
    ax1.annotate(f'MSE: {np.round(mse,3)}',
            xy=(.65, .02), xycoords='axes fraction',fontsize=10)
            # horizontalalignment='left', verticalalignment='top',
            # fontsize=5)
    
    ax2.hist(y_true-y_predicted)
    ax2.set_xlabel(f'Residuals, {var_name}')
    if title != None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()