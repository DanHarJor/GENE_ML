import os
import pickle
class Results():
    def __init__(self, name):
        self.name = name
        self.growthrate_predict_seen = None
        self.growthrate_predict_unseen = None

        if not os.path.exists('saved_results'):
            os.mkdir('saved_results')
        self.path = os.path.join('saved_results',name)
    
    def save(self):
        pickle.dump(self, open(self.path, 'wb'))

    def load(self):
        self = pickle.load(open(self.path, 'rb'))