from scipy.interpolate import interp1d
import numpy as np

class MixingLength():
    def __init__(self,dataset):
        if len(dataset.scanlog_df.columns) > 3:
            raise ValueError("Daniel Says: the scanlog_df must only have 3 columns as we are looking at a ky scan for a single point in the space")
        

        ky, unique_indicies = np.unique(dataset.x.flatten(), return_index=True) # unique also sorts the array
        growthrates = dataset.growthrates[unique_indicies]

        # #sort via ky
        # sorted_indicies = np.argsort(ky)
        # ky = ky[sorted_indicies]
        # unique_indicies = np.argunique(ky)
        # growthrates = growthrates[sorted_indicies]

        self.interpolation = interp1d(ky, growthrates, kind='linear')

    def __call__(self, *args):
        return self.get_label(*args)

    def get_label(self, x):
        return self.interpolation(x)
    



        