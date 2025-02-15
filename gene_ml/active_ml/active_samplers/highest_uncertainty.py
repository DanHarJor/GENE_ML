import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
class HighestUncertainty():
    def __init__(self, function, regressor, random_state=42):
        self.function = function
        self.regressor = regressor # must have a fit, predict function. predict must return uncertainty
        self.integrals = []
        self.rmse_s = []
        self.integrals_dif = None

    def generate_samples(self, num_samples, pool_X, initial_X, initial_y, test_mode=False, integration_interval=None, plot_interval=None):
        self.regressor.fit(initial_X, initial_y)
        X = initial_X
        y = initial_y
        
        if test_mode:
            X_test = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
            y_test = self.function(X_test).reshape(-1,1) #caution function might be heavy
            integral_true = trapz(y_test.flatten(), X_test.flatten()) 
        
        for i in range(1,num_samples+1):
            _, uncertainty = self.regressor.predict(pool_X, return_std=True)
            X_new = pool_X[np.argmax(uncertainty)].reshape(-1,1)
            X = np.concatenate([X, X_new], axis=0)
            y_new = self.function(X_new)
            y = np.concatenate([y, y_new], axis=0)

            self.regressor.fit(X, y)
            if test_mode:
                y_pred_test, uncertainty = self.regressor.predict(X_test, return_std=True)
                self.rmse_s.append(np.sqrt(np.mean((y_pred_test.flatten()-y_test.flatten())**2)))
        
                if integration_interval != None and i%integration_interval==0:
                    self.integrals.append(trapz(y_pred_test.flatten(), X_test.flatten()))

                if plot_interval!=None and i%plot_interval==0:
                    figure, ax = plt.subplots(1,1)
                    ax.plot(X_test, y_pred_test, c='green')
                    ax.fill_between(X_test.flatten(), y_pred_test-uncertainty, y_pred_test+uncertainty, color = [0.1,1,0.1,0.2])
                    ax.scatter(X_test, y_test, c='lightblue')
                    ax.scatter(X, y, c='orange')
                    ax.scatter(X_new, y_new, c='red')
                    # ax.plot(X_test, y_pred_test, c='black')
                    ax.set_title(f'Number of Points: {len(X)}')
                    plt.show()
        self.integrals_dif = np.abs(np.array(self.integrals)-integral_true)
        if test_mode and plot_interval != None:
            fig, ax = plt.subplots(1,1)
            iterations = np.arange(1,num_samples+1)
            ax.plot(iterations, self.rmse_s, label='RMSE', c='red')
            ax_t = ax.twinx()
            ax_t.plot(iterations, self.integrals_dif, label='Integral Difference (right axis)', c='green')
            fig.legend()
            fig.show()   



        