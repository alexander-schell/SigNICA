# -*- coding: utf-8 -*-
"""
@author: Alexander
"""

##########################################
# Differentiable Implementation of SigIC #
##########################################

# For a fully contextualised description of the following code, see ....  

import numpy as np
import scipy
import scipy.linalg as la
from scipy import stats
import pickle
import time
import itertools
from itertools import groupby #to select same-letter subwords; required for barSigCF()
import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import signatory
import MonomialDiscordance as mondis
import SigIC as sigic


'''
This file collects the functions derived in the jupyter notebook .... 
'''

def sha(u, v):
    '''input: u,v : words to be shuffled [strings], n: maximal number of summands [int]
       returns: [u\sha v]_n [list of strings], i.e. the 'first' n summands of the shuffle product of u and v.
       NB: This function is only valid over alphabets consisting of single strings (e.g. i = 10 is a word of length 2)!
    ''' 
    if len(u)*len(v)!=0:  #check if one of the factors is the empty word
        shuff_uv = [w + v[-1] for w in sha(u, v[:-1])] + [w + u[-1] for w in sha(u[:-1], v)] 
        return shuff_uv
    else:
        return [u+v]

    
def lextolin(w, d):
    '''input: word w \in [9]* [string of characters from {1,...,d}], dimension d [=1,...,9].
       returns: the ordinal number [int] of w w.r.t. the 'semi-lexicographical' order (1) on [d]*.
    '''
    k = len(w)
    if k==0:
        return 0
    else:
        a = np.asarray(list(map(int, list(w)))) #convert word to list of its letters (preserving their word-order)
        b = np.asarray([d**(k-(j+1)) for j in range(k)])
        return np.sum(a*b)  # = psi(w) (according to formula (2) above)
    

def tracespring(u,l):
    '''input: word u whose trace is of interest [string with chars '1',...,'9'], 
              maximal length l of offspring words [int]
       returns: a list of all words in <u>* which are of length l'''
    
    tru       = set(u)       # determine trace of u
    tru       = ''.join(tru) # join all letters in the trace of u to one string
    truStar_l = [''.join(x) for x in itertools.product(tru, repeat=l)] # generate all length-l-words in <u>*
    
    return truStar_l  


def crosscumu(u,v,k):
    '''returns: a list of all (homogeneous; there are not others) degree-k polynomials 
                contained in $\W_{u,v}$ [list of lists]'''
    
    W_k_uv    = [] #list of all desired word-polynomials
    # 'i + j = k'
    for i in range(1,k): #go over all 1\leq i \leq k-1, with i = |\tilde{u}| (cf. (A))
        
        u_til = tracespring(u,i)     # generate \tilde{u} and 
        v_til = tracespring(v, k-i)  # \tilde{v} as in (A)
        
        for ut in u_til:
            for vt in v_til:
                q = sha(ut,vt)       # generate u_t\shuffle v_t (word-polynomial => ~list of strings)
                W_k_uv.append(q)
        
    return W_k_uv


def deg_poly(q):
    '''input: word polynomial q [list of strings]
       returns: the degree of q (ie, length of the longest word in q) [int]
    '''
    return len(max(q, key=len)) # returns length of the longest string in q

def d_m(d,m):
    return sum([d**i for i in range(1,m+1)])

def m_dm(dm,d):
    '''Computes number of summands m [int] from the value dm of the geometric sum d_m defined above.'''
    m = np.log(dm*(d-1) + d)/np.log(d) - 1 #the formula for the inversion d_m --> m follows from the geometric sum.
    return int(round(m))


def coeffextract(t,q,d):
    '''input: tensor in $\mathbb{R}^d_{[k]}$, indexed according to (9) [pytorch tensor of shape (N',)], 
              'coefficient polynomial' q [list of strings], d [dimension of X; int] 
       returns: coefficient <t,q> (cf. (7)) [double]. 
    '''
    # Check whether depth of t is at least as large as deg(q):
    d_ESX     = m_dm(len(t), d) # depth of esig_X
    m         = deg_poly(q)     # deg of q
    if m > d_ESX:
        print('The provided tensor is of insufficient depth.')
        return
    
    # Compute c:=<t,q>, using that <t,q> = t_{\psi(q)} (cf. (9), for \psi the linear extension of (2))
    c         = 0
    for w in q:
        c     += t[lextolin(w,d)-1] #gives zero-dimensional torch tensor; coefficients are retrieved according to (9)
                                                                          # (subtract 1 because of Python-indexation)
    return c.item()   #returns the numerical value of c  


def ret_uv(k):
    '''input: k [int]
       returns: the words u ='12...(k-1)' and v='k' [strings].''' # Example: ret_uv(5) returns ('1234', '5')
    u = ''
    v = str(k)
    j = 1
    while len(u)<(k-1):
        u += str(j)
        j += 1
    return u,v

def coeffMats(mu):
    '''Input: global control parameter mu (torch.tensor of shape (d-1,)); case-string s [string]
       Returns: coefficient-extractor matrices [C_1, C_2, C_3] as defined in (\gamma)'''
    
    d      = len(mu) + 1    #get dimension (= number of channels)
    dm     = d_m(d,max(mu)) #get the depth that is required of the input tensor
    
    # Create lists W={W_k, k=2,...,d} of cross-polynomials (cf. (6)/(4)) whose assoc. cumulants (7)/(6) we want to compute:
    W      = [[]]*(d-1)
    
    for k in range(d-1):
        u,v       = ret_uv(k+2)                   #set u=u_k = '12...(k-1)', v=v_k = 'k' for k=2,...,d
        for j in range(int(round(mu[k]))):
            W[k] = W[k] + crosscumu(u,v,j+1)      #assemble W_k(u_k,v_k) = \sqcup_{j=1,...,mu[k]} crosscumu(u,v,j) (cf. (4))
        
    # Get number of elements in W ($= m_\phi$):
    m_phi  = sum([len(w) for w in W]) 
    
    # Compute Matrix C_1 (cf. (\alpha),(\gamma)):
    C1     = torch.zeros(m_phi,dm)
    i      = 0            # row index of C1 (~ enumeration of W)
    for k in range(d-1):
        for q in W[k]:
            for j in range(dm):
                C1[i,j-1] = int(j in [lextolin(w,d) for w in q])  #'-1' because of Python-indexation
            i += 1
            
    # Compute Matrix C_2 (cf. (\alpha),(\gamma)):
    C2     = torch.zeros(d,dm)
    for i in range(d):
        psi_ii = lextolin(2*str(i+1),d) 
        C2[i,psi_ii-1] = 1  #'-1' because of Python-indexation
    
    # Compute Matrix C_3 (cf. (\alpha),(\gamma)):
    C3     = torch.zeros(m_phi,d)                              
    i      = 0            # row index of C2 (~ enumeration of W)
    for k in range(d-1):
        for q in W[k]:
            for j in range(d):
                C3[i,j] = q[0].count(str(j+1))/2 #number of times the letter 'j+1' appears in the word q[0] 
            i +=1
            
    return C1,C2,C3

def phi_mu(t):
    '''Input: t [torch.tensor of shape (d_m,)] ; requires C1,C2,C3 = coeffMats(mu) as auxiliary structures.
       Returns: \phi_\mu(t) as given by (\beta).'''
    
    q1 = torch.matmul(C1,t)
    q2 = torch.exp(torch.matmul(C3,torch.log(torch.matmul(C2,t))))
    q3 = torch.div(q1,q2)
    c  = torch.sum(torch.pow(q3,2))
    
    return c

'''"Epsilon-Robustified" Variants of \phi_mu:'''
     
def phimu(t,epsilon):
    '''Input: t [torch.tensor of shape (d_m,)] ; requires C1,C2,C3 = coeffMats(mu) as auxiliary structures.
       Returns: \phi_\mu(t) as given by (\beta).''' 
    
    q1 = torch.mv(C1,t)
    qa = torch.mv(C2,t)
    qb = torch.log(qa)
    qc = torch.mv(C3,qb)
    q2 = torch.exp(qc)
    q3 = torch.div(q1,q2+epsilon) #'ad-hoc regularization': adding small epsilon in denominator to prevent nans
    c  = torch.sum(torch.pow(q3,2))    
    
    return c

def SigICLoss(x,target,epsilon):
    '''Input: collection of estimated (semi-lexic. indexed) signature coefficients S [torch.tensor of shape (N,d_m)];
              requires control tuple mu and coeff-matrices C1,C2,C3 = coeffMats(mu) as auxiliary structures.
       Returns: normalised cumulant-based contrast value barSigCF_mu(NN_theta(X)) (~ (6) on SigContrast.ipynb).'''
    
    #d     = mu.size + 1    # get dimension (=number of channels of X)
    #m     = max(mu)

    # Compute ESig from x=NN(X):
    [N,d] = x.shape        
    Np    = N//winlen      
    x     = x[:Np*winlen] 
    xsp   = torch.split(x,winlen) 
    xsp   = torch.stack(xsp) 
    sigx  = signatory.signature(xsp,m)

    ESigX = torch.mean(sigx,0)    # estimate expected signature as row-wise arithmetic mean of input 
    ESigX = ESigX.unsqueeze(0)  # add extra dimension (signatory() requires input to be 2d, dims corr.to (batch,channels))
    
    # Compute log(ESig) and apply loss:
    logES = signatory.signature_to_logsignature(ESigX,d,m,mode='expand') #--> torch.tensor of shape(N',), ind.~(9)
    logES = logES.squeeze(0) #reduce superfluous first dimension (which was just needed for signatory())
    
    l     = phimu(logES,epsilon)
    
    return l 













