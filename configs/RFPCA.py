
import numpy as np 

#K = [5, 10, 15] # number of features
# K = [3,5,7] # number of features
K = np.arange(2, 22, 2)

hyper_params = dict(
    T_test = 1000, ##number of iterations for each group
    K = K,
    alpha = 0.15,
    lamda = 0.5,
    retrial = 20 # number of retrial sub_gradient algorithm
) 

