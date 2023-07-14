import numpy as np

class READ_CURVES:
    def __init__(self, file):
        curve = np.loadtxt(file)
        self.x = curve[:,0]
        self.y = curve[:,1]

