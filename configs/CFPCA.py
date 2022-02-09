
import numpy as np 

#K = [5, 10, 15] # number of features
# K = [3,5,7] # number of features
K = [3]

hyper_params = dict(
    K = K,
    delta = [0., 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
    mu = 0.01,
    dualize = True,
    model_name = ['LinCon']
)