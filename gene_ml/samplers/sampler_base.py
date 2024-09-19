
from scipy.spatial import ConvexHull
class Sampler():
    def __init__(self, samples_array_norm):
        self.samples_array_norm = samples_array_norm

    def hull_volume_norm(self):
        # returns 1 if the points convex hull volume is equal to the hypercube volume
        # a unit hypercube has all sides length 1 and a volume of 1. 
        # returns 0.5 if the points convex hull is half the volume of the hypercube
        convex_hull = ConvexHull(self.samples_array_norm)
        return convex_hull.volume
