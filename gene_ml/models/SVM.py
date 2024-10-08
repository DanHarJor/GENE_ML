from .base import Model
from sklearn import svm 

class SVM(Model):
    def __init__(self):
        self.regressor = svm.SVR()
        print('SVM KERNEL:', self.regressor.kernel)
    def train(self, x, y):
        self.regressor.fit(x, y)

    def fit(self, *args, **kargs):
        self.train(*args, **kargs)
    
    def predict(self, x):
        # should return a prediction and its errors if available
        return self.regressor.predict(x)
    
