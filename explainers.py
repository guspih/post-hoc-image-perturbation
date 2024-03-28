import torch
import itertools
import shap
import numpy as np
import scipy.special
import time
import warnings
from skimage.segmentation import slic
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from PIL import Image
import numbers


#Samplers
def random_sampler(M, sample_size=None, p=0.5):
    '''
    Creates an array of samples where each feature is include (set to 1) with a
    given probability and to be perturbed (set to 0) otherwise.
    Args:
        M (int): Nr of features in each sample that can be perturbed
        sample_size (int): Nr of different samples to generate
        p (float): Probability in [0,1] that a feature is included a sample
    Returns (array): [sample_size, M] array indicating the features to perturb
    '''
    if sample_size is None:
        sample_size = M**2
    return np.random.choice(2, (sample_size, M), p=(1-p, p))

def naive_ciu_sampler(M, sample_size=None, inverse=False):
    '''
    Creates an array of all possible samples where only a single feature is
    indicated to be perturbed (set to 0) or, if inverse, is indicated to not be
    perturbed (set to 1), and adds the sample where no feature is perturbed.
    Args:
        M (int): Nr of features in each sample that can be perturbed
        sample_size (int): Nr of different samples to generate (Has to be M+1)
        inverse (bool): Whether all features but one should be set to 0 (True)
    Returns (array): [M+1,M] array indicating the features to perturb
    '''
    if sample_size is None:
        sample_size = M+1
    elif sample_size != M+1:
        raise ValueError(
            'naive_ciu_sampler can currently only works with samples_size=M+1'
        )
    point = 1 if inverse else 0
    if inverse:
        samples = np.zeros((M, M))
    else:
        samples = np.ones((M, M))
    samples[range(M), range(M)] = point
    samples = np.concatenate((samples, np.ones((1,M))))
    return samples.astype(int)

def shap_sampler(M, sample_size=None, ignore_warnings=False):
    '''
    Creates an array of samples indicating which features to perturb (0) and
    which to include (1) of a given size. Will first create all samples with
    all values the same, then all with a single feature included/perturbed,
    then all with two feature included/perturbed, and so on.
    Args:
        M (int): Nr of features in each sample that can be perturbed
        sample_size (int): Nr of different samples to generate
        ignore_warnings (bool): Ignores unbalanced sample_size warnings if True
    Returns (array): [sample_size, M] array indicating the features to perturb
    '''
    if sample_size is None:
        sample_size = 2**M
    #if sample_size < M and not ignore_warnings:
    if sample_size < M+2 and not ignore_warnings:
        warnings.warn(
            f'WARNING: shap_sampler does not cover all features if sample_size'
            f' < M+2, but sample_size={sample_size} and M={M} was given.'
        )
    samples = np.zeros((sample_size, M), dtype=int)
    
    i=0 # Indicates which sample to write to
    l=0 # Only used to give warning messages
    for r in range(M+1):
        r = r//2 if r%2==0 else M-((r-1)//2)
    #TODO: Figure out how to handle 0 and M features
    #for r in range(2,M+1):
    #    r = r//2 if r%2==0 else M-((r-1)//2) 
        comb = itertools.combinations(range(M), r=r)
        for idx in comb:
            if i == sample_size:
                if not ignore_warnings:
                    warnings.warn(
                        f'WARNING: sample_size={sample_size} for M={M} '
                        f'features gives some features more samples than '
                        f'others. Nearest balanced sample_sizes are {l} and '
                        f'{i+len(list(comb))+1}.'
                    )
                break
            samples[i,idx] = 1
            i += 1
        if i == sample_size:
            break
        l = i
    return samples


# Attributers (Explainers that calculates attribution)
def original_ciu_values(Y, Z, inverse=False, expected_util=0.5):
    '''
    Calculates CIU values without intermediate concepts according to the
    py.ciu.image package. Inverse importance is primarily used with inverse
    inverse sampling (see naive_ciu_sampler). Expected utility is used in
    influence calculation to determine which features have positive influence.
    Args:RISE attribution values of each feature
        Y (array): [N] array of all model values for the perturbed inputs
        Z (array): [N,M] array indicating which features were perturbed (0)
        inverse (bool): Whether to calculate importance as 1-importance
        expected_util (float/[float]]): Sets the baseline for influence>0
    Returns (array, array, array, array):
        [M] array with the contextual importance of each feature
        [M] array with the contextual utility of each feature
        [M] array with the influence (ci*(cu-E[cu])) of each feature
        [M,M] array indicating which feature each attribution scores belong to
    '''
    M = Z.shape[1]
    point = 1 if inverse else 0
    true_y = Y[np.where(np.all(Z==np.ones(M), axis=-1))]
    min_importance = np.zeros(M)
    max_importance = np.zeros(M)
    min_importance[:] = true_y
    max_importance[:] = true_y
    for y, z in zip(Y,Z):
        point_position = z==point
        min_importance[point_position & (min_importance>y)] = y
        max_importance[point_position & (max_importance<y)] = y
    importance = (max_importance-min_importance)
    utility = (true_y - min_importance)/importance
    importance = importance/(np.max(Y)-np.min(Y))
    if inverse:
        importance = 1-importance
    influence = importance*(utility-expected_util)
    return importance, utility, influence, np.eye(M)

def ciu_values(Y, Z, expected_util=0.5, return_samples=None):
    '''
    Calculates CIU values for each feature combination in return_samples or
    each feature combination in Z if return_samples is None. Importance of a
    feature combination is the maximum difference in Y between the inputs where
    only some or none of the given features have been perturbed. Utility of a
    feature combination is the difference between Y of the original inputs and
    the minumum Y where the inputs where only some or none of the given
    features have been perturbed. Influence is the utility subtracted by the
    expected utility and then scaled by the importance.
    Args:
        Y (array): [N] array of all model values for the perturbed inputs
        Z (array): [N,M] array indicating which features were perturbed (0)
        expected_util (float/[float]]): Sets the baseline for influence>0
        return_samples (array): [X,M] array indicating features for attribution
    Returns (array, array, array, array):
        [X] array with the contextual importance of given feature combinations
        [X] array with the contextual utility of given feature combinations
        [X] array with the influence (ci*(cu-E[cu])) of feature combinations
        [X,M] array indicating which feature combination attributions belong to
    '''
    M = Z.shape[1]
    if return_samples is None:
        unique_Z = np.unique(Z, axis=0)
    else:
        unique_Z = return_samples
    true_y = Y[np.where(np.all(Z==np.ones(M), axis=-1))]
    min_importance = np.zeros(unique_Z.shape[0])
    max_importance = np.zeros(unique_Z.shape[0])
    min_importance[:] = 10000
    max_importance[:] = -10000
    for y, z0 in zip(Y,Z):
        for i, z1 in enumerate(unique_Z):
            if np.all(z1*z0 == z1):
                if min_importance[i] > y: min_importance[i] = y
                if max_importance[i] < y: max_importance[i] = y
    importance = (max_importance-min_importance)
    utility = (true_y - min_importance)/importance
    importance = importance/(np.max(Y)-np.min(Y))
    influence = importance*(utility-expected_util)
    return importance, utility, influence, unique_Z

def shap_values(Y, Z):
    '''
    Estimates the shapley value attribution of each feature using KernelSHAP.
    The sum of the SHAP values and base value gives the Y value where nothing
    is perturbed. Each SHAP value measures how much the Y value changes if that
    feature is included (as opposed to if it is perturbed).
    Args:
        Y (array): [N] array of all model values for the perturbed inputs
        Z (array): [N,M] array indicating which features were perturbed (0)
    Returns (array, float, array):
        [M] array with the SHAP values of each feature
        The SHAP base value, approx. the value if all features are perturbed
        [M,M] array indicating which feature each attribution scores belong to
    '''
    M = Z.shape[1]
    S = Z.sum(axis=1)
    
    Z = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1)
    S_vals = np.unique(S)

    kernel_vals = np.sqrt(shap_kernel(M, S_vals))
    sqrt_pis = np.zeros(np.max(S_vals)+1)
    sqrt_pis[S_vals] = kernel_vals
    sqrt_pis = sqrt_pis[S]
    shap = np.linalg.lstsq(sqrt_pis[:, None] * Z, sqrt_pis * Y, rcond=None)
    return shap[0][:-1], shap[0][-1], np.eye(M)

def rise_values(Y, Z):
    '''
    Calculates the attribution as is done in the RISE method. Each feature is
    attributed with the average Y value when that feature is not perturbed. To
    recreate full RISE the samples have to be random (see random_sampler).
    Args:
        Y (array): [N] array of all model values for the perturbed inputs
        Z (array): [N,M] array indicating which features were perturbed (0)
    Returns (array, float, array):
        [M] array with the RISE attribution values of each feature
        [M,M] array indicating which feature each attribution scores belong to
    '''
    importance = np.sum(Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))), axis=0)
    occurance = np.sum(Z, axis=0)
    return importance/occurance, np.eye(Z.shape[1])

def lstsq_lime_values(Y, Z):
    '''
    Calculates attribution of each feature with LIME by fitting a linear
    surrogate model with the least squares method. The attribution of each
    feature is their weight in the linear surrogate.
    Args:
        Y (array): [N] array of all model values for the perturbed inputs
        Z (array): [N,M] array indicating which features were perturbed (0)
    Returns (array, float, array):
        [M] array with the attribution of each feature as the surrogate weights
        [M,M] array indicating which feature each attribution scores belong to
    '''
    return np.linalg.lstsq(Z, Y, rcond=None)[0][:-1], np.eye(Z.shape[1])


# Explainer utils
def shap_kernel(M, s):
    '''
    Hepler function for KernelSHAP that calculates the weight of a sample using
    the nr of features (M) and the nr of included features in each sample (s).
    Args:
        M (int): The number of features in the sample
        s (array): [N] array of the number of included features in each sample
    Returns (array): [N] array witht the SHAP kernel values for each s given M
    '''
    return np.nan_to_num(
        (M - 1) / (scipy.special.binom(M, s) * s * (M - s)),
        posinf=100000, neginf=100000
    )