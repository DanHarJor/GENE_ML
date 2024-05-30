import os
import pickle
class Results():
    def __init__(self, name):
        self.name = name
        self.growthrate_predict_seen = None
        self.growthrate_predict_unseen = None
        self.growthrate_predict_seen_errors = None
        self.growthrate_predict_unseen_errors = None
        

        if not os.path.exists('saved_results'):
            os.mkdir('saved_results')
        self.path = os.path.join('saved_results',name)
    
    def save(self):
        pickle.dump(self, open(self.path, 'wb'))

    def load(self):
<<<<<<< HEAD
        self = pickle.load(open(self.path, 'rb'))
=======
        return pickle.load(open(self.path, 'rb'))
>>>>>>> 1888b3942dc73f27963cb309f269ffc5daa885a7
