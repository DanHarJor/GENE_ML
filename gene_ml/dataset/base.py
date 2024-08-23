import pickle

class DataSet():
    def __init__(self):
        self.x = None
        self.growthrates = None
        self.frequencies = None
        self.run_time = None
    
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
    
    def load(self, path):
        pickle.load(open(path, 'rb'))
        

