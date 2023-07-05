#!/usr/bin/env python

import numpy as np
from scipy import stats as spstats

def simulate(n, p, nclass, nfactor,
        std = 4.0,
        rholist = None, sample_groups = None,
        do_shift_mean = True):

    if sample_groups is None:
        sample_dict = get_sample_indices(nclass, n, shuffle = False)
        sample_groups = [x for k, x in sample_dict.items()]

    if rholist is None:
        rholist = [np.random.uniform(0.7, 0.95) for _ in range(nclass)]

    F = spstats.ortho_group.rvs(p)[:, :nfactor]
    L_full = get_blockdiag_features(1000, n, rholist, sample_groups, rho_bg = 0.0, seed = 2000).T
    L = L_full[:, :nfactor]

    mean = np.zeros((1, p))
    if do_shift_mean:
        mean = np.random.normal(0, 10, size = (1, p))

    noise_std = np.random.normal(0, std, p)
    noise_mean = np.zeros(p)
    noise = np.random.multivariate_normal(noise_mean, np.diag(np.square(noise_std)), size = n)

    Y_true = L @ F.T
    Y = Y_true + mean + noise

    return Y, Y_true, L, F, mean, noise_std


def do_standardize(Z, axis = 0, center = True, scale = True):
    '''
    Standardize (divide by standard deviation)
    and/or center (subtract mean) of a given numpy array Z
    
    axis: the direction along which the std / mean is aggregated.
        In other words, this axis is collapsed. For example,
        axis = 0, means the rows will aggregated (collapsed).
        In the output, the mean will be zero and std will be 1
        along the remaining axes.
        For a 2D array (matrix), use axis = 0 for column standardization
        (with mean = 0 and std = 1 along the columns, axis = 1).
        Simularly, use axis = 1 for row standardization
        (with mean = 0 and std = 1 along the rows, axis = 0).
        
    center: whether or not to subtract mean.
    
    scale: whether or not to divide by std.
    '''
    if scale:
        Znew = Z / np.std(Z, axis = axis, keepdims = True)
    else:
        Znew = Z.copy()
        
    if center:
        Znew = Znew - np.mean(Znew, axis = axis, keepdims = True)

    return Znew


def get_equicorr_feature(n, p, rho = 0.8, seed = None, standardize = True):
    '''
    Return a matrix X of size n x p with correlated features.
    The matrix S = X^T X has unit diagonal entries and constant off-diagonal entries rho.
    
    '''
    if seed is not None: np.random.seed(seed)
    iidx = np.random.normal(size = (n , p))
    comR = np.random.normal(size = (n , 1))
    x    = comR * np.sqrt(rho) + iidx * np.sqrt(1 - rho)

    # standardize if required
    if standardize:
        x = do_standardize(x)

    return x


def get_blockdiag_features(n, p, rholist, groups, rho_bg = 0.0, seed = None, standardize = True):
    '''
    Return a matrix X of size n x p with correlated features.
    The matrix S = X^T X has unit diagonal entries and 
    k blocks of matrices, whose off-diagonal entries 
    are specified by elements of `rholist`.
    
    rholist: list of floats, specifying the correlation within each block
    groups: list of integer arrays, each array contains the indices of the blocks.
    '''
    np.testing.assert_equal(len(rholist), len(groups))
    
    if seed is not None: np.random.seed(seed)
    iidx = get_equicorr_feature(n, p, rho = rho_bg)

    # number of blocks
    k = len(rholist)
    
    # zero initialize
    x = iidx.copy() #np.zeros_like(iidx)
    
    for rho, grp in zip(rholist, groups):
        comR = np.random.normal(size = (n, 1))
        x[:, grp] = np.sqrt(rho) * comR + np.sqrt(1 - rho) * iidx[:, grp]

    # standardize if required
    if standardize:
        x = do_standardize(x)

    return x


def get_blockdiag_matrix(n, rholist, groups):
    R = np.ones((n, n))

    for i, (idx, rho) in enumerate(zip(groups, rholist)):
        nblock = idx.shape[0]
        xblock = np.ones((nblock, nblock)) * rho
        R[np.ix_(idx, idx)] = xblock
        
    return R


def get_sample_indices(nclass, n, shuffle = True):
    '''
    Distribute the samples in the categories (classes)
    '''
    rs = 0.6 * np.random.rand(nclass) + 0.2 # random sample from [0.2, 0.8)
    z = np.array(np.round((rs / np.sum(rs)) * n), dtype = int)
    z[-1] = n - np.sum(z[:-1])
    tidx = np.arange(n)
    if shuffle:
        np.random.shuffle(tidx)
    bins = np.zeros(nclass + 1, dtype = int)
    bins[1:] = np.cumsum(z)
    sdict = {i : np.sort(tidx[bins[i]:bins[i+1]]) for i in range(nclass)}
    return sdict


def reduce_dimension_svd(X, k = None):
    if k is None: k = int(X.shape[1] / 10)
    k = max(1, k)
    U, S, Vt = np.linalg.svd(X)
    Uk = U[:, :k]
    Sk = S[:k]
    return Uk @ Sk
