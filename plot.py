import numpy as np
from functions import compute_errors
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from pathlib import Path
from RFPCA import _fit_RFPCA
from pareto import _fit_PCA
from FairPCA import _fit_FairPCA

np.random.seed(10)

def orthorgonal_groups():
    A_info = (np.array([0., 0.]), np.array([[4., 0.], [0., 0.2]]), 500)
    B_info = (np.array([0., 0.]), np.array([[0.2, 0.4], [0.4, 3.]]), 200)
    A = np.random.multivariate_normal(*A_info)
    B = np.random.multivariate_normal(*B_info)

    X = np.concatenate([A, B], axis=0)
    S0 = np.zeros(A.shape[0])
    S1 = np.ones(B.shape[0])
    S = np.concatenate([S0, S1])
    return X, S, A, B


def eigvec_illustration(result_dir):
    k = 1
    X, S, A, B  = orthorgonal_groups()
    # X, S, A, B  = circle_groups()
    plt.figure(figsize = (10,10), dpi = 300)
    matplotlib.rcParams.update({'font.size': 18})

    plt.scatter(A[:, 0], A[:, 1], marker='o', s=8, label='GroupA')
    plt.scatter(B[:, 0], B[:, 1], marker='^', s=8, label='GroupB')

    params = {'K': k, 'd': X.shape[1]}
    V, ProjMat = _fit_PCA(X, A, B, S, None, params)
    plt.quiver(0, 0, V[0], V[1], color="r", angles='xy', scale_units='xy', scale=0.16, label=f'PCA')

    params = {'K': k, 'd': X.shape[1], 'eta': 0.1, 'T': 1000, 'nEta': 1}
    V, ProjMat = _fit_FairPCA(X, A, B, S, None, params)
    if V[0] > 0: V = - V 
    plt.quiver(0, 0, V[0], V[1], color='g', angles='xy', scale_units='xy', scale=0.16, label=f'Samadi et al. (2018)')

    colors = ["#332288","#0077BB","#44AF99","#009988", "#999933","#DDCC77","#CC6677","#882255","#AA4499", "#EE7733", "#EE3377"]

    for i, lamda in enumerate([0., 0.1, 0.15, 0.17, 0.2, 0.22, 0.23, 0.24, 0.25, 1.0]):
        params = {'K': k, 'd': X.shape[1],
                    'alpha': 0., 'lamda': lamda,
                    'retrial': 40, 'T_test': 1000}
        V, ProjMat = _fit_RFPCA(X, A, B, S, S, params)   
        if V[0] > 0: V = - V 
        ABtr = compute_errors(X, A, B, S, 
                                None, None, None, 
                                None, None, None, 
                                ProjMat, {'ABtrain'})['ABtrain']
        plt.quiver(0, 0, V[0], V[1], color=colors[i], angles='xy', scale_units='xy', scale=0.195, width=0.006)

    lgnd = plt.legend(fontsize=20, loc='lower right')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    filename = result_dir / f'eigenvectors.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"saved to {filename}")
    plt.close()

result_dir = Path('visualize')
result_dir.mkdir(parents=True, exist_ok=True)

exp_dir = result_dir / 'synthetic'
exp_dir.mkdir(parents=True, exist_ok=True)

eigvec_illustration(exp_dir)
