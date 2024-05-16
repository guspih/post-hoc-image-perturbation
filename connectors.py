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

from image_segmenters import perturbation_masks

class SegmentationAttribuitionPipeline():
    def __init__(
        self, segmenter, sampler, perturber, explainer, mask_transform=None,
        per_pixel=False
    ):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.explainer = explainer
        self.mask_transform = mask_transform
        self.per_pixel = per_pixel

        # Variables to hold information between calls
        self.masks = None
        self.transformed_masks = None
        self.samples = None
        self.ys = None
    
    def __call__(self, image, model, **kwargs):
        segments, self.masks = self.segmenter(image)
        if not self.mask_transform is None:
            self.transformed_masks = self.mask_transform(self.masks)
        else:
            self.transformed_masks = self.masks
        self.samples = self.sampler(
            self.masks.shape[0], sample_size=kwargs.get('sample_size')
        )
        distortion_masks = perturbation_masks(
            self.transformed_masks, self.samples
        )
        perturbed_images, perturbed_samples = self.perturber(
            image, distortion_masks, self.samples
        )
        self.ys = model(perturbed_images)
        if len(self.ys.shape)==1:
            self.ys = np.expand_dims(self.ys, axis=-1)
        ret = [self.explainer(y, perturbed_samples) for y in self.ys.T]
        if self.per_pixel:
            ret = [perturbation_masks(
                    values[-2], self.transformed_masks
                ) for values in ret]
        return ret