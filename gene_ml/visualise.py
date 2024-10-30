from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np

# def residual_plot(ax, fig, y_true, y_predicted, var_name, title=None, y_pred_err=None):
#     mse=np.mean((y_true-y_predicted)**2)
#     cb = ax.scatter(y_true, y_predicted, marker='.', c=y_pred_err, cmap='summer')
#     if type(y_pred_err)!=type(None):
#         ax.errorbar(y_true, y_predicted, yerr=y_pred_err, fmt='.', ecolor='red')
#         cb = ax.scatter(y_true, y_predicted, marker='.', c=y_pred_err, cmap='summer', zorder=10)
#         fig.colorbar(cb, ax=ax)
        
#     else:
#         ax.scatter(y_true, y_predicted, marker='.')
     

#     print('MAX',np.max(y_true),np.max(y_predicted))
#     ref = np.linspace(np.min((np.min(y_true),np.min(y_predicted))),np.max((np.max(y_true),np.max(y_predicted))), 100).astype(float)
#     print(np.max(ref))
#     ax.plot(ref,ref, 'r')
#     ax.set_xlabel(f'True {var_name}')
#     ax.set_ylabel(f'Predicted {var_name}')
#     if title != None:
#         ax.set_title(title)
#     ax.annotate(f'MSE: {Decimal(mse):.2E}',
#             xy=(.6, .02), xycoords='axes fraction',fontsize=10)

def residual_plot(ax, fig, y_true, y_predicted, var_name, title=None, y_pred_err=None, font_size=None):
    residuals = y_true - y_predicted
    mse=np.mean((y_true-y_predicted)**2)
    mape = np.mean(np.abs(y_true-y_predicted)*100/y_true)
    if type(y_pred_err)!=type(None):
        ax.errorbar(y_true, residuals, yerr=y_pred_err, fmt='.', ecolor='red')
        # cb = ax.scatter(y_true, residuals, marker='.', c=y_pred_err, cmap='summer', zorder=10)
        # fig.colorbar(cb, ax=ax)
        
    else:
        print('NO UNCERTAINTY PROVIDED')
        ax.scatter(y_true, residuals, marker='.')
     
    # minimum = np.min(y_true)
    # maximum = np.max(y_true)
    # refx = np.linspace(minimum, maximum, 100)#np.linspace(np.min((np.min(y_true),np.min(y_predicted))),np.max((np.max(y_true),np.max(y_predicted))), 100).astype(float)
    # refy = np.zeros(len(refx))
    # ax.plot(refx,refy, 'r')
    ax.axhline(0, 0, 1, color='r')
    # ax.set_xlabel(f'GENE {var_name} (ground truth)', fontsize=font_size)
    ax.set_ylabel(f'Residuals, {var_name}', fontsize=font_size)
    if title != None:
        ax.set_title(f'{title}', fontsize=font_size)##: {len(y_true)} data points')
    # ax.annotate(f'MSE: {Decimal(mse):.2E}%',
    #         xy=(.7, .02), xycoords='axes fraction', fontsize=font_size)
    ax.annotate(f'MAPE: {np.round(mape,1)}%',
            xy=(.7, .02), xycoords='axes fraction', fontsize=font_size)
    


    ax.set_xlim(left=0)

def residual_hist(ax, fig, y_true, y_predicted, var_name, title=None, bins=50, orientation='horizontal'):
    ax.hist(y_true-y_predicted, density=True, bins=bins, orientation=orientation)
    if orientation == 'horizontal':
        None
        # ax.set_ylabel(f'Residuals, {var_name}')
        # ax.set_xlabel(f'Normalised Frequency')
    if orientation == 'verticle':
        ax.set_xlabel(f'Residuals, {var_name}')
        ax.set_ylabel(f'Normalised Frequency')

    if title != None:
        ax.set_title(title)

