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


class RandomSampler():
    '''
    Creates an array of samples where each feature is include (set to 1) with a
    given probability and to be perturbed (set to 0) otherwise.
    Args:
        p (float): Probability in [0,1] that a feature is included a sample
    '''
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: [sample_size, M] array indicating the features to perturb
        '''
        if sample_size is None:
            sample_size = M**2
        return np.random.choice(2, (sample_size, M), p=(1-self.p, self.p))

class SingleFeatureSampler():
    '''
    Creates an array of all possible samples where only a single feature is
    indicated to be perturbed (set to 0) or, if inverse, is indicated to not be
    perturbed (set to 1). Optionally, adds the samples where all/no features
    are perturbed.
    Args:
        inverse (bool): Whether all features but one should be set to 0 (True)
        add_all (bool): If True, adds a sample where all features are perturbed
        add_none (bool): If True, adds a sample where no features are perturbed
    '''
    def __init__(self, inverse=False, add_all=False, add_none=False):
        self.inverse = inverse
        self.add_all = add_all
        self.add_none = add_none

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate (=M+1)
        Returns: [M+1,M] array indicating the features to perturb
        '''
        if sample_size is None:
            sample_size = M
        elif sample_size != M:
            raise ValueError(
                'SingleFeatureSampler currently only works with samples_size=M'
            )
        point = 1 if self.inverse else 0
        if self.inverse:
            samples = np.zeros((M, M))
        else:
            samples = np.ones((M, M))
        samples[range(M), range(M)] = point
        if self.add_all:
            samples = np.concatenate((samples, np.zeros((1,M))))
        if self.add_none:
            samples = np.concatenate((samples, np.ones((1,M))))
        return samples.astype(int)

class ShapSampler():
    '''
    Creates an array of samples indicating which features to perturb (0) and
    which to include (1) of a given size. Will first create all samples with
    all values the same, then all with a single feature included/perturbed,
    then all with two feature included/perturbed, and so on.
    Args:
        ignore_warnings (bool): Ignores unbalanced sample_size warnings if True
    '''
    def __init__(self, ignore_warning=False):
        self.ignore_warnings = ignore_warning

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: [sample_size, M] array indicating the features to perturb
        '''
        if sample_size is None:
            sample_size = 2**M
        #if sample_size < M and not ignore_warnings:
        if sample_size < M+2 and not self.ignore_warnings:
            warnings.warn(
                f'WARNING: shap_sampler does not cover all features if '
                f'sample_size < M+2, but sample_size={sample_size} and M={M} '
                f'was given.'
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
                    if not self.ignore_warnings:
                        warnings.warn(
                            f'WARNING: sample_size={sample_size} for M={M} '
                            f'features gives some features more samples than '
                            f'others. Nearest balanced sample_sizes are {l} '
                            f'and {i+len(list(comb))+1}.'
                        )
                    break
                samples[i,idx] = 1
                i += 1
            if i == sample_size:
                break
            l = i
        return samples
