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

from image_perturbers import SingleColorPerturber, ReplaceImagePerturber, cast_image
from image_segmenters import perturbation_masks


def auc_sampler(vals, sample_size=None, deletion=False):
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
        sample_size = vals.size+1
    elif sample_size > vals.size+1:
        raise ValueError(
            f'sample_size must be <= vals.size+1, but '
            f'sample_size={sample_size} and vals.size={vals.size}.'
        )
    indices = np.argsort(vals)[::-1]
    point = 1
    if deletion:
        point = 0
        samples = np.ones((sample_size, vals.size))
    else:
        point = 1
        samples = np.zeros((sample_size, vals.size))
    old_j=0
    for n, i in enumerate(np.linspace(0,vals.size,sample_size)):
        j = round(i)
        idxs = indices[old_j:j]
        samples[n:, idxs] = point
        old_j = j
    return samples

class ImageAUCEvaluator():
    def __init__(self, mode='srg', perturber=None, return_curves=False):
        self.delete = []
        if mode != 'deletion':
            self.delete.append(False)
        if mode != 'insertion':
            self.delete.append(True)
        self.mode = mode
        self.perturber = perturber
        self.return_curves = return_curves
        if perturber is None:
            self.perturber = SingleColorPerturber((190,190,190))

    def __call__(self, image, masks, model, vals, output_idxs=..., **kwargs):
        scores = []
        curves = []
        for delete in self.delete:
            samples = auc_sampler(vals, kwargs.get('sample_size'), delete)
            distortion_masks = perturbation_masks(masks, samples)
            perturbed_images, perturbed_samples = self.perturber(
                image, distortion_masks, samples
            )
            ys = model(perturbed_images)
            if len(ys.shape)==1:
                ys = np.expand_dims(ys, axis=-1)
            curves.append(ys[output_idxs])
            scores.append(np.mean(ys, axis=0)[output_idxs])
        if self.mode == 'srg':
            scores.append(scores[0]-scores[1])
        if self.return_curves:
            return scores, curves
        return scores

