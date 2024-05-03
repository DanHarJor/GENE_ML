from matplotlib import pyplot as plt
import numpy as np

def residuals(y_true, y_predicted, var_name, title=None):
    ncol = 2
    fig, [ax1, ax2] = plt.subplots(1,ncol, figsize=(3*ncol,3))
    ax1.scatter(y_true, y_predicted)
    ref = np.linspace(0,np.max((np.max(y_true),np.max(y_predicted))), 100).astype(int)
    ax1.plot(ref,ref, 'g')
    ax1.set_xlabel(f'True {var_name}')
    ax1.set_ylabel(f'Predicted {var_name}')
    
    ax2.hist(y_true-y_predicted)
    ax2.set_xlabel(f'Residuals, {var_name}')
    if title != None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()