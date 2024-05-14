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
import cv2

from image_perturbers import SingleColorPerturber, ReplaceImagePerturber, cast_image
from image_segmenters import perturbation_masks

class TopVisualizer():
    def __init__(self, k=None, p=None, treshold=None, perturber=None):
        self.perturber = perturber
        if perturber is None:
            self.perturber = SingleColorPerturber((190,190,190))
        if (k is None) + (p is None) + (treshold is None) != 2:
            raise ValueError('Exactly one of k, p, and treshold should be set')
        self.k = k
        self.p = p
        self.treshold = treshold
        if not k is None:
            self.mode = 'k'
        elif not p is None:
            self.mode = 'p'
        else:
            self.mode = 'treshold'

    def __call__(self, values, image, segment_masks, pixel_mask=None):
        nr_segments = values.shape[0]
        to_perturb = np.ones((1,nr_segments))
        if self.mode != 'threshold':
            if self.mode == 'p':
                self.k = int(np.ceil(self.p*nr_segments))
            top_order = values.argsort()
            to_perturb[0, top_order[:-self.k]] = 0
        else:
            to_perturb[0, values>self.treshold] = 0
        distortion_mask = perturbation_masks(segment_masks, to_perturb)
        perturbed_image = self.perturber(image, distortion_mask, to_perturb)[0]
        return perturbed_image[0]

class HeatmapVisualizer():
    def __init__(
            self, normalize=False, image_weight=0.5, colormap=cv2.COLORMAP_JET,
            invert_colormap=False
        ):
        self.perturber = ReplaceImagePerturber(replace_images=None)
        self.normalize = normalize
        self.image_weight = image_weight
        self.colormap = colormap
        self.invert_colormap = invert_colormap

    def __call__(self, values, image, segment_masks, pixel_mask=None):
        if pixel_mask is None:
            heatmap = perturbation_masks(segment_masks, values.reshape((1,-1)))
        else:
            heatmap = pixel_mask
        if len(heatmap.shape) == 3:
            heatmap = heatmap[0]
        if self.normalize:
            heatmap = heatmap-np.min(heatmap)
            heatmap = heatmap/np.max(heatmap)
        if self.invert_colormap:
            heatmap = 1-heatmap
        heatmap = cast_image(heatmap, np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap=self.colormap)
        perturbed_image = self.perturber(
            image, np.full(image.shape[:1], self.image_weight), None,
            replace_images=heatmap
        )[0]
        return perturbed_image[0]