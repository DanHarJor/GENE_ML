from sklearn import svm 

class SVM():
    def __init__(self):
        self.regressor = svm.SVR()
    def train(self, x, y):
        self.regressor.fit(x, y)
    
    def predict(self, x):
        return self.regressor.predict(x)
    