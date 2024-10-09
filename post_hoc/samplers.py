import itertools
import numpy as np
import random

import warnings

class RandomSampler():
    '''
    Creates an array of samples where each feature is included (=1) with a
    given probability and to be perturbed (=0) otherwise.

    Args:
        p (float): Probability in [0,1] that a feature is included a sample
    '''
    def __init__(self, p=0.5):
        self.p = p
        self.deteministic = False

    def __str__(self):
        return f'RandomSampler({self.p})'

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: 
            array: [sample_size, M] index of the features to perturb per sample
        '''
        if sample_size is None:
            sample_size = M
        return np.random.choice(2, (sample_size, M), p=(1-self.p, self.p))

class UniqueRandomSampler():
    '''
    Creates an array of samples where each feature is included (=1) or perturbed
    (=0) with equal probability. Samples in such a way that no two samples are
    the same.
    '''
    def __str__(self):
        return f'UniqueRandomSampler()'

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: 
            array: [sample_size, M] index of the features to perturb per sample
        '''
        if sample_size is None:
            sample_size = M
        elif sample_size > 2**M:
            sample_size = 2**M
        if M < 61:
            samples = random.sample(range(2**M), sample_size)
            samples = [list('{:b}'.format(n).rjust(M,'0')) for n in samples]
            return np.array(samples, dtype=int)
        samples = np.random.choice(2, (sample_size, M))
        s = random.sample(range(2**61), sample_size)
        s = np.array(
            [list('{:b}'.format(n).rjust(61,'0')) for n in s], dtype=int
        )
        samples[:, :61] = s
        return samples

class SampleProbabilitySampler():
    '''
    Creates and array of samples where each feature is set to be perturbed (=0)
    or not to be perturbed (=1) with a probability that, for each sample, is
    drawn from a given distribution. Available distributions are 'uniform',
    'normal' (truncated), and 'beta'. The distribution probabilities can be 
    inversed.

    Args:
        distribution (str): The distribution to draw p for each sample from
        inverse (bool): If True, samples are drawn using 1-p instead
        **kwargs: Additional arguments for the distributions
    '''
    def __init__(self, distribution='uniform', inverse=False, **kwargs):
        self.deteministic = False
        self.distribution = distribution
        self.inverse = inverse
        self.kwargs = kwargs
        distributions = ['uniform', 'normal', 'beta']
        if not distribution in distributions:
            raise ValueError(
                f'distribution={distribution} should be one of {distributions}'
            )
        from scipy.stats import truncnorm
        self.truncnorm = truncnorm

    def __str__(self):
        kw = ','.join(np.sort([f'{k}={self.kwargs[k]}' for k in self.kwargs]))
        kw = ',' + kw if len(kw) > 0 else kw
        content = f'{self.distribution},{self.inverse}{kw}'
        return f'SampleProbabilitySampler({content})'
    
    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: 
            array: [sample_size, M] index of the features to perturb per sample
        '''
        if sample_size is None:
            sample_size = M
        if self.distribution == 'uniform':
            sample_p = np.random.rand(sample_size,1)
        elif self.distribution == 'normal':
            loc = self.kwargs.get('loc', 0.5)
            scale = self.kwargs.get('scale', 1)
            a, b = (0-loc)/scale, (1-loc)/scale
            sample_p = self.truncnorm(a, b, loc, scale).rvs(
                size=(sample_size,1)
            )
        elif self.distribution == 'beta':
            a = self.kwargs.get('a', 1.5)
            b = self.kwargs.get('b', 1.5)
            sample_p = np.random.beta(a, b, size=(sample_size,1))
        if self.inverse:
            sample_p = 1-sample_p
        return (sample_p < np.random.rand(sample_size,M)).astype(np.int32)

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
        self.deterministic = True
    
    def __str__(self):
        content = f'{self.inverse},{self.add_all},{self.add_none}'
        return f'SingleFeatureSampler({content})'

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate (=M+1)
        Returns:
            array: [M(+1/+2),M] index of the features to perturb per sample
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
        inverse (bool): Whether to order the many perturbations first instead
        ignore_warnings (bool): Ignores unbalanced sample_size warnings if True
    '''
    def __init__(self, inverse=False, ignore_warning=False):
        self.inverse = inverse
        self.ignore_warnings = ignore_warning
        self.deterministic = True
    
    def __str__(self):
        return f'ShapSampler()'

    def __call__(self, M, sample_size=None):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: 
            array: [sample_size, M] index of the features to perturb per sample
        '''
        if sample_size is None:
            sample_size = M+2
        #if sample_size < M and not ignore_warnings:
        if sample_size < M+2 and not self.ignore_warnings:
            warnings.warn(
                f'WARNING: shap_sampler does not cover all features if '
                f'sample_size < M+2, but sample_size={sample_size} and M={M} '
                f'was given.'
            )
        if sample_size > 2**M:
            sample_size = 2**M
        if self.inverse:
            point = 1
            samples = np.zeros((sample_size, M), dtype=int)
        else:
            point = 0
            samples = np.ones((sample_size, M), dtype=int)
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
                samples[i,idx] = point
                i += 1
            if i == sample_size:
                break
            l = i
        return samples

class MultiSampler():
    '''
    Creates an array of samples indicating which features to perturb (0) and
    which to include (1) of a given size. Samples from a given set of samplers
    and samples from them according to a defined split size.

    Args:
        samplers ([callable]): The samplers to sample from
        array: [X] The fraction of samples to draw from each of the X samplers
    '''
    def __init__(self, samplers, split=None):
        self.samplers = samplers
        if split is None:
            self.split = np.full(len(samplers), 1/len(samplers))
        else:
            self.split = split/np.sum(split)

    def __str__(self):
        content = f'[{",".join(self.samplers)}],[{",".join(self.split)}]'
        return f'MultiSampler({content})'

    def __call__(self, M, sample_size):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns: 
            array: [sample_size, M] index of the features to perturb per sample
        '''
        float_sizes = self.split*sample_size
        sizes = np.floor(float_sizes).astype(int)
        missing = sample_size-np.sum(sizes)
        remainers = float_sizes-sizes
        sizes[np.argpartition(remainers, missing)[:missing]] += 1
        samples = []
        for sampler, size in zip(self.samplers, sizes):
            samples.append(sampler(M, size))
        return(np.concatenate(samples))

class AllNoneWrapperSampler():
    '''
    Creates an array of samples indicating which features to perturb (0) and
    which to include (1) of a given size. Samples are taken from a given
    sampler and the indicated samples of all or no features being perturbed is
    appended.

    Args:
        add_all (bool): If True, adds a sample where all features are perturbed
        add_none (bool): If True, adds a sample where no features are perturbed
    '''
    def __init__(self, sampler, add_all=False, add_none=False):
        if not (add_all or add_none):
            raise ValueError('One or both of add_all and add_none should True')
        self.sampler = sampler
        self.add_all = add_all
        self.add_none = add_none
    
    def __str__(self):
        content = f'{self.sampler},{self.add_all},{self.add_none}'
        return f'AllNoneWrapperSampler({content})'

    def __call__(self, M, sample_size):
        '''
        Args:
            M (int): Nr of features in each sample that can be perturbed
            sample_size (int): Nr of different samples to generate
        Returns:
            array: [sample_size(+1/+2),M] index of the features to perturb
        '''
        samples = self.sampler(M, sample_size)
        if self.add_all:
            samples = np.concatenate((samples, np.zeros((1,M))))
        if self.add_none:
            samples = np.concatenate((samples, np.ones((1,M))))
        return samples