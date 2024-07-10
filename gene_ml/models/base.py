import pickle
import os
class Model():
    def __init__(self, name):
        self.parent = 'parent'
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        self.path = os.path.join('saved_models',name)
    
    def save(self, path=None):
        if path==None:
            path = self.path
        pickle.dump(self, open(path, 'wb'))

    
    def load(self,path=None):
        if path==None:
            path= self.path
        return pickle.load(open(path, 'rb'))

    