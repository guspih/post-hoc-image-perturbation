import torch
import itertools
import shap
import numpy as np
import scipy.special
import time
import warnings


def shap_kernel(M, s):
    #return ((M-1)/(
    #    ((M+1).lgamma()-(s+1).lgamma()-((M-s)+1).lgamma()).exp()*s*(M-s)
    #)).nan_to_num(posinf=100000, neginf=100000)
    return np.nan_to_num(
        (M - 1) / (scipy.special.binom(M, s) * s * (M - s)),
        posinf=100000, neginf=100000
    )

def shap_values(Y, Z):
    M = Z.shape[1]#torch.tensor(Z.size(dim=1))
    S = Z.sum(axis=1)
    
    Z = np.concatenate((Z, torch.ones((Z.shape[0], 1))), axis=1) #torch.cat((Z, torch.ones((Z.size(0), 1))), dim=1)
    S_vals = np.unique(S)

    sqrt_pis = np.sqrt(shap_kernel(M, S_vals))
    test = np.zeros(np.max(S_vals)+1)
    test[S_vals] = sqrt_pis
    sqrt_pis = test[S]
    #return torch.linalg.lstsq(sqrt_pis.unsqueeze(dim=1) * Z, sqrt_pis * Y)
    return np.linalg.lstsq(sqrt_pis[:, None] * Z, sqrt_pis * Y, rcond=None)

def shap_sampler(M, sample_size=None, ignore_warnings=False):
    if sample_size is None:
        sample_size = 2**M
    #if sample_size < M and not ignore_warnings:
    if sample_size < M+2 and not ignore_warnings:
        warnings.warn(
            f'WARNING: shap_sampler does not cover all features if sample_size'
            f' < M+2, but sample_size={sample_size} and M={M} was given.'
        )
    #samples = torch.zeros((sample_size, M), dtype=int)
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
                        f'WARNING: sample_size={sample_size} for M={M} features '
                        f'gives some features more samples than others. Nearest '
                        f'balanced sample_sizes are {l} and {i+len(list(comb))+1}.'
                    )
                break
            samples[i,idx] = 1
            i += 1
        if i == sample_size:
            break
        l = i
    return samples



def test_shap(f, x, reference, sample_size=None):
    M = x.shape[0]
    samples = shap_sampler(M, sample_size=sample_size)
    #X = reference.tile(dims=(samples.shape[0],1))
    X = np.tile(reference, reps=(samples.shape[0],1))
    #x = x.tile(dims=(samples.shape[0],1))
    x = np.tile(x, reps=(samples.shape[0],1))
    idxs = samples==1
    X[idxs] = x[idxs]
    #y = torch.tensor(f(X), dtype=torch.float32)
    y = f(X)
    return shap_values(y, samples)

def f(X):
    np.random.seed(0)
    beta = np.random.rand(X.shape[-1])
    return np.dot(X, beta) + 10


M = 4
np.random.seed(1)
x = np.random.randn(M)
reference = np.zeros(M)



tic = time.perf_counter()   
#a = test_shap(f, torch.tensor(x, dtype=torch.float32), torch.tensor(reference, dtype=torch.float32))
a = test_shap(f, x, reference, sample_size=6)
toc = time.perf_counter()
print(f'my took {toc - tic:0.4f} seconds')

print('a', a[0][:-1], a[0][-1])


explainer = shap.KernelExplainer(f, np.reshape(reference, (1, len(reference))))
shap_values = explainer.shap_values(x, nsamples=4)
print("shap_values =", shap_values)
print("base value =", explainer.expected_value)