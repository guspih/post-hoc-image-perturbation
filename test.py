import torch
import itertools
import shap
import numpy as np
import scipy.special
import time
import warnings
from skimage.segmentation import slic
from skimage.transform import resize
from PIL import Image
import numbers

def shap_kernel(M, s):
    #return ((M-1)/(
    #    ((M+1).lgamma()-(s+1).lgamma()-((M-s)+1).lgamma()).exp()*s*(M-s)
    #)).nan_to_num(posinf=100000, neginf=100000)
    return np.nan_to_num(
        (M - 1) / (scipy.special.binom(M, s) * s * (M - s)),
        posinf=100000, neginf=100000
    )

def naive_ciu_values(Y, Z):
    M = Z.shape[1]
    C = np.array([[[100000.0,-100000.0],[100000.0,-100000.0]]]*M)
    for y, z in zip(Y,Z):
        for i, c in enumerate(C):
            c[z[i]] = (min(c[z[i]][0], y), max(c[z[i]][1], y))
    importance = np.zeros(M)
    for i, c in enumerate(C):
        importance[i] = max(abs(c[0,1]-c[1,0]), abs(c[0,0]-c[1,1]))
    return importance

def original_ciu_values(Y, Z, inverse=False, expected_util=0.5):
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
    return importance, utility, influence

def ciu_values(Y, Z):
    pass

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

def rise_values(Y, Z):
    importance = np.sum(Z*(Y.reshape(list(Y.shape)+[1]*(Z.ndim-1))), axis=0)
    occurance = np.sum(Z, axis=0)
    return importance/occurance
    #return np.sum(Z*Y[:,None], axis=0)/np.sum(Z, axis=0)


def random_sampler(M, sample_size=None, p=0.5):
    if sample_size is None:
        sample_size = M**2
    return np.random.choice(2, (sample_size, M), p=(1-p, p))

def naive_ciu_sampler(M, sample_size=None, inverse=False):
    if sample_size is None:
        sample_size = M
    point = 1 if inverse else 0
    samples = np.ones((sample_size, M)) if not inverse else np.zeros((sample_size, M))
    samples[range(M), range(M)] = point
    samples = np.concatenate((samples, np.ones((1,M))))
    return samples.astype(int)

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


def perturbation_masks(segment_masks, samples):
    return np.tensordot(samples, segment_masks, axes=(1,0))

def single_color_pertuber(image, segment_masks, samples, color):
    #TODO: Decide if the input should be segment_masks or perturbation masks
    if not issubclass(image.dtype.type, numbers.Integral):
        if isinstance(color[0], numbers.Integral):
            color = np.array(color)/255

    sample_masks = perturbation_masks(segment_masks, samples) # TODO: Consider moving pertubation masking outside the pertubation itself
    #perturbed_segments = np.tile(image, (sample_masks.shape[0],1,1,1))#perturbation_masks*image
    #perturbed_segments[sample_masks==0] = color
    perturbed_segments = np.tile(image-color, (sample_masks.shape[0],1,1,1))
    perturbed_segments = (
        perturbed_segments*sample_masks.reshape(list(sample_masks.shape)+[1])
    )+color
    if isinstance(color[0], numbers.Integral):
        perturbed_segments = perturbed_segments.astype(int, copy=False)
    return perturbed_segments, samples

def replace_image_perturbation(image, replace_image, sample_masks, samples):
    perturbed = np.tile(image-replace_image, (sample_masks.shape[0],1,1,1))
    perturbed = (
        perturbed*sample_masks.reshape(list(sample_masks.shape)+[1])
    )+replace_image
    if issubclass(image.dtype.type, numbers.Integral):
        perturbed = perturbed.astype(int, copy=False)
    return perturbed, samples

def transform_pertubation(image, transform, sample_masks, samples):
    return replace_image_perturbation(
        image, transform(image), sample_masks, samples
    )


def slic_segmenter(image, nbr_segments, compactness):
    segments = slic(
        image,
        n_segments=nbr_segments,
        compactness=compactness,
        start_label=0,
    )
    segment_ids = np.unique(segments)
    segment_masks = np.zeros((segment_ids.shape[0],segments.shape[0], segments.shape[1]))
    for id in segment_ids:
        segment_masks[id] = np.where(segments==id,1.0,0.0)
    return segments, segment_masks

def grid_segmenter(image, h_segments, v_segments, rise_upscaling=False):
    h_size, v_size = image.shape[0:2]
    segments = np.arange(h_segments*v_segments).reshape(h_segments, v_segments)
    segments = resize(
        segments, (h_size,v_size), order=0, mode='reflect', anti_aliasing=False
    )

    segment_masks = np.zeros((h_segments*v_segments, h_segments, v_segments))
    for i in range(h_segments*v_segments):
        segment_masks[i, i//h_segments, i%v_segments] = 1
    segment_masks = resize(
        segment_masks, (h_segments*v_segments,h_size,v_size),
        order=1 if rise_upscaling else 0, mode='reflect', anti_aliasing=False
    )
    
    return segments, segment_masks