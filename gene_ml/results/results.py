import os
import pickle
class Results():
    def __init__(self, name, *args, **kargs):
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
        return pickle.load(open(self.path, 'rb'))
    
    def exists(self):
        return os.path.exists(self.path)
    
class ResultsUQ(Results):
    def __init__(self, name):
        super().__init__(name)
        self.growthrate_nominal = None
        self.growthrate_lower_bound = None
        self.growthrate_upper_bound = None
        self.growthrate_predict_uqsamples = None
        self.bounds = None
        self.nominal_parameters = None

if __name__ == '__main__':
    ruq = ResultsUQ(name='test_name')

    ruq.growthrate_predict_seen = 5
    ruq.nominal_growthrate = 3

    print(ruq.growthrate_predict_seen, ruq.nominal_growthrate)
