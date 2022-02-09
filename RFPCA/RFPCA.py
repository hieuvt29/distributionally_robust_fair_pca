# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:31:37 2021

@author: Kam-Fung (Henry) Cheung

Algorithm 1 RFPCA Solver
"""

import numpy as np
import scipy.linalg as la
from joblib import Parallel, delayed


class RFPCA:
    """ Robust Fair Principal Component Analysis """
    
    def __init__(
            self,
            eps=np.array([0.1, 0.2]),
            data=None,
            sensitives=None,
            k = 1,
            lamda = 0.1,
            T = 1000,
            retrial = 20,
    ):
        self.X = data # Sensitive attributes should not be included
        self.S = sensitives
        self.eps = eps
        self.N = data.shape[0] # assume each datum is stored in a row
        assert len(self.S) == self.N
        self.k = k # no. of features
        self.d = data.shape[1] # total no. of features
        self.lamda = lamda
        
        # Below parameters are for the algorithm
        self.T = T                  # number of iterations
        self.retrial = retrial      # number of random restart
        
        # self.M_all = self.X.T@self.X/self.N
    
        # M0 and M1 are *conditional* sample average of the second moment matrix
        # M0 and M1 in Theorem (2.4)
        M0_idx = np.arange(self.N)[self.S == 0]
        M1_idx = np.arange(self.N)[self.S == 1]
        self.M0 = 1./ len(M0_idx) * self.X[M0_idx].T @ self.X[M0_idx]
        self.M1 = 1./ len(M1_idx) * self.X[M1_idx].T @ self.X[M1_idx]
        self.M = np.array([self.M0, self.M1])
        
        self.get_marginals()        
        self.check_conditions()
        self.compute_constants()
        
        # Principal components and its orthogonal
        self.components_ = 0
        self.orthocomponents_ = 0
        
        self.error = False
        if self.k >= self.d:
        #if self.k > self.d:
            print('Warning! k has to be smaller than d.')
            self.error = True
    
    def get_marginals(self):
        """Calculate marginal probabilities of data"""
        self.P_1 = np.sum(self.S == 1)/self.N
        self.P_0 = np.sum(self.S == 0)/self.N

        if np.abs(self.P_1 + self.P_0 - 1) > 1e-10:
            print(np.abs(self.P_1 + self.P_0 - 1))
            print('Marginals are WRONG!')

    def compute_constants(self):
        """Compute the constants kappa, theta, vartheta"""
        # Equation (9c)
        self.kappa = np.array([(self.P_0+self.lamda)*self.eps[0]+(self.P_1-self.lamda)*self.eps[1], 
                               (self.P_1+self.lamda)*self.eps[1]+(self.P_0-self.lamda)*self.eps[0]])
        self.theta = np.array([2*np.abs(self.P_0+self.lamda)*np.sqrt(self.eps[0]),
                               2*np.abs(self.P_1+self.lamda)*np.sqrt(self.eps[1])])
        self.vartheta = np.array([2*np.abs(self.P_0-self.lamda)*np.sqrt(self.eps[0]), 
                                  2*np.abs(self.P_1-self.lamda)*np.sqrt(self.eps[1])])
        self.C = np.array([(self.P_0+self.lamda)*self.M0 + (self.P_1-self.lamda)*self.M1,
                           (self.P_1+self.lamda)*self.M1 + (self.P_0-self.lamda)*self.M0])
    
    def check_conditions(self):
        if self.lamda == np.inf:
            self.lamda = np.minimum(self.P_0, self.P_1)/2
            print(f" set lambda = {self.lamda}")

        if self.lamda > np.minimum(self.P_0, self.P_1):
            # Compute the eigenvalues. (sorted in ascending order)
            w0 = la.eigh(self.M[0], eigvals_only=True)
            w1 = la.eigh(self.M[1], eigvals_only=True)

            # check the sum of the (d-k) smallest eigenvalues
            if np.sum(w0[0:(self.d-self.k)]) < self.eps[0] \
            or np.sum(w1[0:(self.d-self.k)]) < self.eps[1]:                
                print('Conditions not met. Resetting lamda to minimum value.')
                self.lamda = np.minimum(self.P_0, self.P_1)
        return True
    
    def F(self, U, a): 
        UUT = U@U.T
        # Objective function (in Formulation (11) with a = 0 or 1)
        res =  self.kappa[a] + self.theta[a]*np.sqrt(np.trace(UUT@self.M[a])) + \
            self.vartheta[1-a]*np.sqrt(np.trace(UUT@self.M[1-a])) +\
                np.trace(UUT@self.C[a])
        return res
    
    def loss(self, U, M): 
        # Calculate the reconstruction error
        # U: current iterate U
        # M: sample average of the conditional second moment
        # Notice that there is no need to divide by the number of samples
        return np.trace(U@U.T@M)

    def penalty(self,U, M0, M1): 
        #(empirical) unfairness measure
        #out = abs(loss(U,X0)-loss(U,X1))
        # equation below Theorem 1.2
        return np.abs(self.loss(U,M0) - self.loss(U,M1))        

    def gradF(self, U, a): 
        # This function computes the Riemannian gradient
        #Riemann gradient function, i.e. Equation (12)      
        UUT = U@U.T 
        out1 = np.trace(UUT@self.M[a])
        out2 = np.trace(UUT@self.M[1-a])
        return (np.eye(self.d)-UUT)@((self.theta[a]/np.sqrt(out1))*self.M[a]+\
                                 ((self.vartheta[1-a]/np.sqrt(out2))*self.M[1-a]+2*self.C[a]))@U
        
    ##setting
            
    def retract(self, U, Delta): 
        # polar retraction function onto the Stiefel manifold
        # Input:
        # U: current iterate
        # Delta: direction on the tangent space
        # Equation (13)
        return (U+Delta)@self.sqrtinv(np.eye(self.d-self.k) + np.dot(Delta.T, Delta))
    
    def sqrtinv(self, Sigma):
        # This function computes the square root inverse of a symmetric matrix 
        # Input:
        # Sigma: symmetric, positive definite matrix
        Sigma = np.asarray(Sigma)
        if len(Sigma.shape) != 2:
            raise ValueError("Non-matrix input to matrix function.")
        w, v = la.eigh(Sigma)
        w = 1/np.sqrt(np.maximum(w, 0))
        return (v * w).dot(v.conj().T)

    # Retract to the Stiefel using the qr decomposition of X + G.
    # Copied from pymanopt
    def qr_retract(self, X, G):
        # Calculate 'thin' qr decomposition of X + G
        q, r = np.linalg.qr(X + G)
        # Unflip any flipped signs
        XNew = np.dot(q, np.diag(np.sign(np.sign(np.diag(r)) + 0.5)))        
        return XNew
    
    # Generate random Stiefel point using qr of random normally distributed
    # matrix. Copied from pymanopt
    def rand(self): # for U0 in Algorithm 1
        X = np.random.randn(self.d, self.d - self.k)
        q, r = np.linalg.qr(X)
        return q

    def fit(self):
        # Run algorithm 1 for many random initialization
        F_min = np.inf
        Obj = np.zeros(self.T)
        # initial guess start with M0
        # w, v = la.eigh(self.M0)
        # U0 = v[:, 0:(self.d-self.k)]
        # U, thisF, delta_norms = self.subgradient_descent(U0)
        # F_min = thisF[-1]
        
        # # initial guess start with M1
        # w, v = la.eigh(self.M1)
        # U0 = v[:, 0:(self.d-self.k)]
        # thisU, thisF, delta_norms = self.subgradient_descent(U0)
        # if thisF[-1] < F_min:
        #     U = thisU
        #     Obj = thisF
        #     F_min = thisF[-1]

        # initial guess start with M
        w, v = la.eigh(self.P_0*self.M0 + self.P_1*self.M1)
        U0 = v[:, 0:(self.d-self.k)]
        thisU, thisF, delta_norms = self.subgradient_descent(U0)

        if thisF[-1] < F_min:
            U = thisU
            Obj = thisF
            F_min = thisF[-1]
        
        res = Parallel(n_jobs=10)(delayed(self.subgradient_descent)(self.rand()) for i in range(self.retrial))
        for idx, (thisU, thisF, delta_norms) in enumerate(res):
            if thisF[-1] < F_min:
                U = thisU
                Obj = thisF
                F_min = thisF[-1]

        #w, V= la.eigh(np.eye(self.d)-U@U.T, subset_by_index = [self.d-self.k, self.d-1])
        
        w, V= la.eigh(np.eye(self.d)-U@U.T)
        V = V[:, (self.d-self.k):(self.d)]
        self.components_ = V
        self.orthocomponents_ = U
        return V, U, Obj

    
    def subgradient_descent(self, U0, alpha = 0): # Algorithm 1
        ## T: number of iterations
        ## alpha: index of descent rate, (= 0 in our original case) 
    
        U = U0
        T = self.T
        Obj = np.zeros(T)
        delta_norms = np.zeros(T)
        # set constant step size
        gamma0 = 1/np.sqrt(T+1) ##initial step-size

        for i in range(T):
            gamma = gamma0 ##update gamma(step-size) 

            # Compute the Riemannian gradient
            if(self.F(U, 0) >= self.F(U, 1)): ##arg max{F0(U),F1(U)}
                # index is 0
                Delta = self.gradF(U, 0) 
            else: 
                Delta = self.gradF(U, 1) 

            delta_norms[i] = np.linalg.norm(Delta, 'fro') 
            # Execute the retraction to the Stiefel manifold
            #U = self.retract(U,-gamma*Delta) ##retraction using polar
            U = self.qr_retract(U, - gamma*Delta) ## retraction using qr (copied from manopt)
            Obj[i] = np.maximum(self.F(U, 0), self.F(U, 1))
            
        return U, Obj, delta_norms


def _fit_RFPCA(Xtr, Atr, Btr, Str, Ytr, params):
    alpha = params['alpha']
    lamda = params['lamda']
    retrial = params['retrial']
    T_test = params['T_test']
    k = params['K']
    
    eps = np.array([alpha/np.sqrt(Atr.shape[0]), alpha/np.sqrt(Btr.shape[0])])
    print(f'\r\033[K testing with alpha={alpha}, lamda={lamda}', end="")
    drpca = RFPCA(data = Xtr, k = k, sensitives = Str, lamda = lamda, \
            retrial = retrial, eps = eps, T = T_test)
    V, U, Obj = drpca.fit() ##implement algorithm1 with different lamda
    PrV = V@V.T
    return V, PrV