
import numpy as np 

#K = [5, 10, 15] # number of features
# K = [3,5,7] # number of features
K = [3]

hyper_params = dict(
    K = K,
    delta = 0.0,
    mu = [0.0001, 0.001, 0.01, 0.05, 0.5],
    dualize = True,
    model_name = ['LinCovCon']
)