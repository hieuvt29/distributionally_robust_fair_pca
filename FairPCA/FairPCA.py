import numpy as np
from sklearn.decomposition import PCA
import scipy.linalg as la

def re(Y, Z):
    # Calculate the reconstruction error of matrix Y with respect to matrix Z
    # Matrix Y and Z are of the same size
    reVal = np.linalg.norm(Y-Z, 'fro')**2
    return reVal


def pca(A, d):
    ppca = PCA(n_components=d)
    ppca.fit(A)
    return ppca.components_.T

def oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2):
    
    # Given an input matrix_A= A^T A, matrix_B=B^T B both of size n by n, d, and weights w_1,w_2, solve the
    # optimization problem
    # min w_1 z_1 + w_2 z_2 s.t.
    # z_1 >= alpha - <matrix_A , P>
    # z_2 >= beta - <matrix_B , P>
    # tr(P) <= d
    # 0 <= P <= I

    if A.shape != (m_A,n) or B.shape != (m_B,n): #wrong size
        print('Input matrix to oracle method has wrong size. Set P, l_1, l_2 to be 0')
        P_o = 0
        z_1 = 0
        z_2 = 0

    covA = A.T@A
    covB = B.T@B

    # We weight A^T A by w_1 and B^T B by w_2. Note that A^T A = summation of
    # v_i v_i^T over vector v_i in group A, so w_1 A^T A can be obtained by
    # scaling each v_i to sqrt(w_1) v_i. Similar for group B.
    X_ = np.vstack([(np.sqrt((1/m_A)*w_1))*A, (np.sqrt((1/m_B)*w_2))*B])
    coeff_P_o = pca(X_, d)

    # coeff_P_o is now an n x d matrix
    P_o = coeff_P_o @ coeff_P_o.T
    z_1 = (1/m_A)*(alpha - sum(sum( covA * P_o )))
    z_2 = (1/m_B)*(beta - sum(sum( covB * P_o )))
    return P_o, z_1, z_2

def optApprox(M, d):
    # UNTITLED3 Summary of this function goes here
    #    Detailed explanation goes here
    coeff = pca(M, d)
    P = coeff @ coeff.T
    Mhat = M@P
    return Mhat

def mw(A, B, d, eta, T):

    # matrix A has the points in group A as its rows
    # matrix B has the points in group B as its rows
    # population A and B are expected to be normalized to have mean 0. 
    # d is the target dimension
    # eta and T are MW's parameters

    # print('MW method is called')

    covA = A.T@A
    covB = B.T@B

    # m_A and m_B are size of data set A and B respectively
    m_A = A.shape[0]
    m_B = B.shape[0]
    n = A.shape[1]

    Ahat = optApprox(A, d)
    alpha = np.linalg.norm(Ahat, 'fro')**2

    Bhat = optApprox(B, d)
    beta = np.linalg.norm(Bhat, 'fro')**2

    # MW

    # start with uniform weight
    w_1 = 0.5
    w_2 = 0.5

    # P is our answer, so I keep the sum of all P_t along the way
    P = np.zeros(n)
    # just for record at the end to see the progress over iterates
    record = [["iteration" "w_1" "w_2" "loss A" "loss B" "loss A by average" "loss B by average"]]

    for t in range(1, T+1):
    
        # think of P_temp as P_t we got by weighting with w_1,w_2
        [P_temp,z_1,z_2] = oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)
        
        # z_1, z_2 are losses for group A and B respectively. If z_i is big, group i is
        # bottle neck, so weight group i more next time
        w_1star = w_1*np.exp(eta*z_1)
        w_2star = w_2*np.exp(eta*z_2)
    
        # renormalize
        w_1 = w_1star / (w_1star+w_2star)
        w_2 = w_2star / (w_1star+w_2star)
        
        # add to the sum of P_t
        P = P+P_temp
        
        # record the progress
        P_average = (1/t) * P
        record.append([t, w_1, w_2, z_1, z_2, (1/m_A)*(alpha - sum(sum( covA @ P_average ))), (1/m_B)*(beta - sum(sum( covB @ P_average )))])

    # take average of P_t
    P = (1/T) * P

    # calculate loss of P_average
    z_1 = 1/(m_A)*(alpha - sum(sum(covA@P)))
    z_2 = 1/(m_B)*(beta - sum(sum(covB@P)))
    z = max(z_1,z_2)

    # in case last iterate is preferred to the average
    P_last = P_temp

    # calculate loss of P_average
    zl_1 = 1/(m_A)*(alpha - sum(sum(covA@P_last)))
    zl_2 = 1/(m_B)*(beta - sum(sum(covB@P_last)))
    z_last = max(zl_1,zl_2)

    # print('MW method is finished. The loss for group A is ', str(z_1), 'For group B is ', str(z_2))
    # print(record)
    return P, z, P_last, z_last


   

def _fit_FairPCA(Xtr, Atr, Btr, Str, Ytr, params):
    nEta = params['nEta']
    eta = params['eta']
    T = params['T']
    k = params['K']

    print(f"\033[K\rtesting eta = {eta}", end="")
    # Fair PCA part
    P_fair, z_kkjj, P_last, z_last_kkjj = mw(Atr, Btr, k, eta, T)
    # z[jj] = z_kkjj
    # z_last[jj] = z_last_kkjj
    if z_kkjj < z_last_kkjj:
        P_smart = P_fair
    else:
        P_smart = P_last

    I_P_smart = np.eye(np.size(P_smart,0))-P_smart
    eig, vec = la.eigh(I_P_smart)
    eig[eig < 0] = 0
    eig_sqrt = np.lib.scimath.sqrt(eig)
    assert sum(np.iscomplex(eig_sqrt)) == 0

    I_P_smart_sqrt = vec @ np.diag(eig_sqrt) @ vec.T

    P_smart = np.eye(np.size(P_smart,0)) - I_P_smart_sqrt
    # just done with P smart as my fair PCA solution with equal loss
    w, vecs = la.eigh(P_smart)

    # assert np.alltrue(w[:-k] < 1e-6) 
    V = vecs[:, -k:]
    # assert np.linalg.norm(V@V.T - P_smart, 'fro') < 1e-6
    return V, P_smart
