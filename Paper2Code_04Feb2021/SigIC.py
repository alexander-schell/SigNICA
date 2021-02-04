# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:54:34 2020

@author: Alexander
"""

##############################################
# Implementation of SigIC (Cumulant Contrast)#
##############################################

import numpy as np
import scipy
import scipy.linalg as la
from scipy import stats
import pickle
import itertools
from itertools import groupby
import torch
import signatory


'''
NOTE: The function SigCF() in this file corresponds to the function barSigCF() in SigContrast.ipynb.
For a full documentation of the following functions, see the Jupyter Notebook 'SigContrast.ipynb'.
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





def ESig(X,m,winlen):
    '''input: data X [pytorch tensor of shape (N,d)], tensor level m [int], windowlength of each sample-ts [winlen; int]
       returns: pytorch tensor of shape (N',) containing the values of (8) according to the indexation (9).
    '''
    Y     = torch.clone(X) # copy data
    [N,d] = Y.shape        # collect dimensions of data
    Np    = N//winlen      # N' = \lfloor(N/n)\rfloor
    Y     = Y[:Np*winlen]  # cut data so that it can be 'divided into obvervational batches without remainder'
    
    # Split Y into Np-many chunks of length winlen:
    Ys    = torch.split(Y, winlen) # Ys is a tuple of length Np containing tensors of shape (winlen, d)
    Ys    = torch.stack(Ys)        # Ys is tensor of shape (Np,winlen,d)
    
    # Compute signature (up to depth m) of each \hat{X}_j, j=1,...,Np:
    SigX  = signatory.signature(Ys, m) # tensor of shape (Np, d_m) (with d_m as defined in the text); 
                                       # 'jth-row' of SigX is signature (up to depth m) of batch \hat{X}_j, indexed ~ (9)
    
    # Compute arithmetic mean of rows of SigX (~ (8)):
    ESigX = torch.mean(SigX.double(), 0) # (Convert SigX into a tensor of double-entries so that mean can be computed.)
    
    return ESigX





def deg_poly(q):
    '''input: word polynomial q [list of strings]
       returns: the degree of q (ie, length of the longest word in q) [int]
    '''
    return len(max(q, key=len)) # returns length of the longest string in q

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





def sigcumulextract(X,winlen,W): # (this function is needed for (meta) detail-checks only)
    '''input: data X [pytorch tensor of shape (N,d)], windowlength of sample-ts [winlen; int],
              coefficient-polynomials W [list of list of strings]
       returns: signature cumulants (7) for all q in W [list (of length |W|) of doubles].
    '''
    # Determine largest degree of elements in W:
    m         = max([deg_poly(q) for q in W])
    
    # Compute Expected Signature of X up to depth m: 
    esig_X    = ESig(X,m,winlen)
    esig_X    = esig_X.unsqueeze(0) #signatory needs the tensor esig_X to be 2-dimensional, dims corr. to (batch,channels)
    
    # Compute logarithmic coordinates of ESigX (wrt monomial standard basis of \mathbb{R}^d_{[k]}):
    d         = X.shape[1] #collect number of channels of X
    logEsig_X = signatory.signature_to_logsignature(esig_X, d, m, mode='expand') #--> torch.tensor of shape(N',), ind.~(9)
    logEsig_X = logEsig_X.squeeze(0) #reduce superfluous first dimension (which was just needed for signatory())
    
    # Extract the desired coordinates {<logES(x), q> | q in W}:
    sigcums   = []
    for q in W:
        sigcums.append(coeffextract(logEsig_X,q,d))
    
    return sigcums




def logESig(X,m,winlen):
    '''input: data X [pytorch tensor of shape (N,d)], windowlength of sample-ts [winlen; int], tensor depth m [int]
       return: log-coordinates of ESig(X) up to depth m [pytorch tensor of shape (N',), indexed ~ (9)]
    '''
    # Compute Expected Signature of X up to depth m:
    esig_X    = ESig(X,m,winlen)
    esig_X    = esig_X.unsqueeze(0) #signatory needs the tensor esig_X to be 2-dimensional, dims corr. to (batch,channels)
    
    # Compute logarithmic coordinates of ESigX (wrt monomial standard basis of \mathbb{R}^d_{[k]}):
    d         = X.shape[1] #collect number of channels of X
    logEsig_X = signatory.signature_to_logsignature(esig_X, d, m, mode='expand') #--> torch.tensor of shape(N',), ind.~(9)
    
    return logEsig_X.squeeze(0) #reduce superfluous first dimension (which was just needed for signatory())





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





# Rescale Data to unit amplitude:
def rescale(X):
    '''input: X [pytorch tensor of shape (N,d)]
       returns: rescaled tensor X* = (X[:,i]/(|X[:,i]|.max()) ; i=0,...,d-1)'''
    Y = torch.clone(X)
    for i in range(X.shape[1]):
        Y[:,i] = (1/torch.max(torch.abs(Y[:,i])))*Y[:,i]
    
    return Y




def SigIC(X,mu,winlen):
    '''input: data X [pytorch tensor of shape (N,d)], windowlength of sample-ts [winlen; int] (cf. (8)),
              'control-tuple' mu [np-array of shape (d-1,); entries ordered by number of channel] 
       returns: Phi_mu(X) (sum over mu-controlled subselection of cross-cumulants (6)) [double] 
    '''
    d      = X.shape[1] #number of components of X
    X      = rescale(X) #rescale data to unit amplitude (to achieve scale-invariance of the SigIC-criterion) 
    
    # Create lists W={W_k, k=2,...,d} of cross-polynomials (cf. (6)/(4)) whose assoc. cumulants (7)/(6) we want to compute:
    W      = [[]]*(d-1)
    mu_max = np.zeros(d-1) #allocate to determine length of longest monomial in W
    
    for k in range(d-1):
        u,v       = ret_uv(k+2)                   #set u=u_k = '12...(k-1)', v=v_k = 'k' for k=2,...,d
        for j in range(int(round(mu[k]))):
            W[k] = W[k] + crosscumu(u,v,j+1)      #assemble W_k(u_k,v_k) = \sqcup_{j=1,...,mu[k]} crosscumu(u,v,j) (cf. (4))
        mu_max[k] = max([deg_poly(q) for q in W[k]]) #determine length of longest monomial in W[k]
        
    mu_max = int(round(np.max(mu_max))) #length of longest monomial in W
    
    ### Print W if you want to see over which cross-shuffled elements q the sum (6) is taken:
    #print('W =', W) 
    
    # Compute log-coordinates of ESig(X) up to depth mu_max
    logESX = logESig(X,mu_max,winlen)
    
    s      = np.zeros(d-1)
    # Compute the kth-summand of the outer sum in (6) 
    for k in range(d-1):
        s[k] = sum([coeffextract(logESX,q,d)**2 for q in W[k]]) #k-th inner sum in (6) [cf. (9)]
    
    # Compute \Phi_mu(X) = \sum_{k=0}^{d-2} s[k] (cf. (6))
    SigICX  = np.sum(s)
    
    return SigICX


def SigCF(X,mu,winlen):
    '''input: data X [pytorch tensor of shape (N,d)], windowlength of sample-ts [winlen; int] (cf. (8)),
              'control-tuple' mu [np-array of shape (d-1,); entries ordered by number of channel] 
       returns: \bar{Phi}_mu(X) (sum over mu-controlled subselection of normalized cross-cumulants (13)) [double] 
    '''
    d      = X.shape[1] #number of components of X
    
    # Create lists W={W_k, k=2,...,d} of cross-polynomials (cf. (6)/(4)) whose assoc. cumulants (7)/(6) we want to compute:
    W      = [[]]*(d-1)
    mu_max = np.zeros(d-1) #allocate to determine length of longest monomial in W
    
    for k in range(d-1):
        u,v       = ret_uv(k+2)                   #set u=u_k = '12...(k-1)', v=v_k = 'k' for k=2,...,d
        for j in range(int(round(mu[k]))):
            W[k] = W[k] + crosscumu(u,v,j+1)      #assemble W_k(u_k,v_k) = \sqcup_{j=1,...,mu[k]} crosscumu(u,v,j) (cf. (4))
        mu_max[k] = max([deg_poly(q) for q in W[k]]) #determine length of longest monomial in W[k]
        
    mu_max = int(round(np.max(mu_max))) #length of longest monomial in W
    
    ### Print W if you want to see over which cross-shuffled elements q the sum (6) is taken:
    #print('W =', W) 
    
    # Compute log-coordinates of ESig(X) up to depth mu_max
    logESX = logESig(X,mu_max,winlen)
    
    s      = torch.zeros(d-1)
    # Compute the kth-summand of the outer sum in (13):
    for k in range(d-1):
        # Compute normalization (12) for each q in W[k]:
        norms = dict.fromkeys([tuple(q) for q in W[k]]) #allocate dictionary (with q \in W[k] as keys, i.e.: norms = {q: None}) 
        for q in W[k]:
            ordrep          = ''.join(sorted(q[0])) #returns ordered representative \vec{q} of q
            normfactors_0   = [''.join(g) for _, g in groupby(ordrep)] #select 'monoletteral subwords' of \vec{q}
            # Compute factors according to (12):
            normfactors_1   = torch.tensor([coeffextract(logESX,[f[0]+f[0]],d)**(-len(f)/2) for f in normfactors_0])
            norms[tuple(q)] = torch.prod(normfactors_1) #compute normalisation constant (~ denominator in (12))
        s[k] = torch.sum(torch.tensor([(norms[tuple(q)]*coeffextract(logESX,q,d))**2 for q in W[k]])) #k-th inner sum in (13)
    
    # Compute \Phi_mu(X) = \sum_{k=0}^{d-2} s[k] (cf. (6))
    SigICX  = torch.sum(s)
    
    return SigICX

# Scaling each data-channel to unit amplitude:
def rescale(X):
    '''input: X [pytorch tensor of shape (N,d)]
       returns: rescaled tensor X* = (X[:,i]/(|X[:,i]|.max()) ; i=0,...,d-1)'''
    Y = torch.clone(X)
    for i in range(X.shape[1]):
        Y[:,i] = (1/torch.max(torch.abs(Y[:,i])))*Y[:,i]
    
    return Y

#################################################################################
#################################################################################

def rem_outls(X, gam): #identical to remove_outliers() in SigIC.ipynb
    Y  = torch.clone(X)
    d  = Y.unsqueeze(1).shape[-1]
    m  = torch.mean(Y,0)
    ym = torch.abs(torch.max(Y - m,0)[0])
    for i in range(d):
        Y  = Y[torch.abs(Y[:,i] - m[i].item()) < (1-gam[i])*ym[i]]
    return Y





def SigCFrescale(X,mu,winlen, gamma):
    '''input: data X [pytorch tensor of shape (N,d)]; windowlength of sample-ts [winlen; int] (cf. (8));
              'control-tuple' mu [np-array of shape (d-1,), entries ordered by number of channel],
              'outlier-proportion-tuple' gamma [np-array of shape (d,), gamma[i]=proportion of largest values to delete from X[:,i]] 
       returns: Phi_propro(X) (preprocessed SigIC, as defined in (11)) [double] 
    ''' 
    # Preprocess the data:
    Y = torch.clone(X)
    Y = crop_outliers(Y, gamma) #delete outliers (delete the largest gamma[i]*N values from X[:,i])
    Y = torch.from_numpy(Y)     #convert back to torch tensor
    Y = rescale(Y)              #rescale 'outlier-cleaned' data
    
    # Apply SigIC to preprocessed data:
    s = SigIC(Y,mu,winlen)
    
    return s    


















