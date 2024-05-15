import pickle

class DataSet():
    def __init__(self):
        None
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(self, path):
        pickle.load(open(path, 'rb'))
        

