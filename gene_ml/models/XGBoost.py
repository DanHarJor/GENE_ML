from xgboost import XGBRegressor 
from .base import Model

import numpy as np

class XGBoost(Model):
    def __init__(self):
        self.model_type_id = 'xgb'
        self.regressor = XGBRegressor(objective='reg:squarederror')

    def train(self, x, y):
        self.regressor.fit(x,y)

    def fit(self, *args, **kargs):
        self.train(*args, **kargs)
    
    def tune_hypers(self, x, y):
        raise NotImplemented
    def predict(self, x):
        y_predict = self.regressor.predict(np.array(x))
        return y_predict
    
    def sensitivities(self):
        return self.regressor.get_fscore()
    
if __name__ == '__main__':
    print('hello world')