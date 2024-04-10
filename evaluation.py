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

def auc_sampler(vals, sample_size=None,  deletion=False):
    '''
    Creates an array of samples where each following sample has more inserted
    or deleted samples than the preceeding one. The most influential features,
    as determined by a given attribution array, is inserted or deleted first.
    Args:
        vals (array): The attribution scores of the different features
        sample_size (int): Nr of samples to generate (<=len(vals))
        deletion (bool): If True, samples are steadily deleted, else inserted
    Returns (array): [sample_size, M] array indicating the features to perturb
    '''
    if sample_size is None:
        sample_size = vals.size
    elif sample_size > vals.size:
        raise ValueError(
            f'sample_size must be <= vals.size, but sample_size={sample_size} '
            f'and vals.size={vals.size}.'
        )
    indices = np.argsort(vals)[::-1]
    point = 1
    if deletion:
        point = 0
        samples = np.ones((sample_size, vals.size))
    else:
        point = 1
        samples = np.zeros((sample_size, vals.size))
    step = vals.size/sample_size
    i=0
    old_j=0
    for n in range(sample_size):
        i+=step
        j = round(i)
        idxs = indices[old_j:j]
        samples[n:, idxs] = point
        old_j = j
    return samples