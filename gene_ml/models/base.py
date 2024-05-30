import pickle
<<<<<<< HEAD

class Model():
    def __init__(self):
        None
    
    def save(path):
        pickle.dump(self, open(path, 'wb'))

    
    def load(path):
        pickle.load(self, open(path, 'rb'))
=======
import os
class Model():
    def __init__(self, name):
        self.parent = 'parent'
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        self.path = os.path.join('saved_models',name)
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    
    def load(self):
        return pickle.load(open(self.path, 'rb'))
>>>>>>> 1888b3942dc73f27963cb309f269ffc5daa885a7

    