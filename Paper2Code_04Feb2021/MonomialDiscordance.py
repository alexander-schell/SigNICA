# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:54:34 2020

@author: Alexander
"""

##########################################
# Implementation of Monomial Discordance #
##########################################

import numpy as np
import scipy
import scipy.linalg as la
import matplotlib.pyplot as plt 

from scipy import stats # Needed for Spearman's rho
from itertools import permutations


'''
Implement the monomial discordance of Def. 53 (Sect. 2.3) 
(via batch-wise sample mean estimation (Monte-Carlo)).

Input: R^d-valued time series ("X and Y"), given as (d,N)-arrays
       (N = n*k with n as in Def. 53 and k (=N/n) the number of batches)
Output: Estimate of \varrho(X, Y)
'''

# 1.) Chop time-series into batches and compute the associated concordance matrix 

def ConcordanceMatrix(X_in,Y_in,n,skip,authority):
    ''' Input:  observed time series X,Y as (N',d)-arrays, with N'>=N(:=n*k) and 'window-length' n and 'skip-number'
                 skip (-->(skip-1)-many intermediate windows of length n are discarded) and authority \in \{'spearman', 'kendall'\}
        Output: Concordance Matrix of (X,Y) to the window-length n (as introduced in Def. 53)'''

    X      = X_in.copy().transpose()
    Y      = Y_in.copy().transpose()
    [d, N] = np.shape(X)
    
    # Delete surplus observations to enable an 'even-numbered estimation' of \varrho
    N      = n*(N//n)
    X      = X[:,:N]
    Y      = Y[:,:N]
    
    # Compute Concordance Matrix
    C      = np.zeros((d,d))
    k      = int(N/n)
    for i in range(d):
        # reshape time series of ith component into an array whose jth column gives
        # the realisations for the Spearman-coeff \rho(X^i_j, Y^i_j)
        SpearX = X[i,:].reshape(k,n) # --> (k,n)-array s.t. l-th column is left argument of rho_S(X^i_l,Y^j_l)
        SpearX = SpearX[::skip]      # -->(skip-1)-many intermediate windows of length n are discarded  
        for j in range(d):
            SpearY = Y[j,:].reshape(k,n) # --> (k,n)-array s.t. l-th column is right argument of rho_S(X^i_l,Y^j_l)
            SpearY = SpearY[::skip]      # -->(skip-1)-many intermediate windows of length n are discarded  
            # compute the 'Spearman-summands' that define C_{X,Y}(i,j):
            spears = np.zeros(n)
            if(authority=='spearman'):
                for l in range(n): 
                        spears[l] = np.abs(stats.spearmanr(SpearX[:,l], SpearY[:,l])[0]) 
            else:
                for l in range(n): 
                    spears[l] = np.abs(stats.kendalltau(SpearX[:,l], SpearY[:,l])[0])
            # sum over the 'Spearman-summands' to obtain C_{X,Y}(i,j):
            C[i,j] = (1/n)*np.sum(spears)
            
    return C
        
    # Divide X and Y into k batches of size (d,n)
    #k      = int(N/n)
    #X_b    = np.array_split(X,k,axis=1)
    #Y_b    = np.array_split(Y,k,axis=1)


# 2.) Compute the Monotone Concordance (Def. 53) between X and Y
    
def MonomialDiscordance(X_in,Y_in,n,authority):
    ''' Input:  observed time series X,Y as (N',d)-arrays, with N'>=N(:=n*k);
                authority \in \{'spearman', 'kendall'\}
        Output: Monotone Concordance between X and Y (Def. 53)'''
    
    X      = X_in.copy().transpose()
    Y      = Y_in.copy().transpose()
    [d, N] = np.shape(X)
    
    # Compute Concordance Matrix of (X_in, Y_in):
    if(authority=='spearman'):
        C      = ConcordanceMatrix(X_in,Y_in,n,1,'spearman')
    else:
        C      = ConcordanceMatrix(X_in,Y_in,n,1,'kendall')
        
    # Create list of Permutation Matrices:
    # collect permutations:
    perms   = list(permutations(np.arange(d).tolist())) # can permute lists only
    perMats = [] # initialize list of permutation matrices
    # construct permutation matrices (and store them in perMats):
    E       = np.identity(d)
    for p in perms:
        P   = np.zeros((d,d))
        for i in range(d):
            P[i]=E[np.asarray(p)[i]]
        perMats.append(P)
        
    # Compute minimal (L1)L2-distance of C from elements in perMats:
    mindist = (1/np.sqrt(d))*min([np.linalg.norm(C - P) for P in perMats])
    
    return mindist
     
    
    















