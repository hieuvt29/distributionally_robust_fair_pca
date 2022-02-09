import numpy as np 

nEta = 1
# K = [3,5,7] # number of features
K = np.arange(2, 22, 2)
hyper_params = dict(
    nEta = nEta,
    eta = 0.1, # increment of 0.1
    T = 1000, 
    K = K
)
