import numpy as np

def load_data(filepath):
    '''
    Load in the given csv filepath as a numpy array
    '''
    *X, y = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=True,
        unpack=True,
    )
    X = np.array(X, dtype=float).T 
    return X, y.reshape((-1, 1))
