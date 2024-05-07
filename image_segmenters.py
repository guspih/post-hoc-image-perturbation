import torch
import itertools
import shap
import numpy as np
import scipy.special
import time
import warnings
from skimage.segmentation import slic
from skimage.transform import resize
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from PIL import Image
import numbers
import cv2


# Image segmenters
class WrapperSegmenter():
    '''
    Segments an image using a given segmentation function that returns a [H,W]
    array and then splits that array into a mask per segment.
    Args:
        segmenter (callable): Callable segmenter with image as 1st argument
        **kwargs: Additional arguments to pass to the segmenter
    '''
    def __init__(self, segmenter, **kwargs):
        self.segmenter = segmenter
        self.kwargs = kwargs
    
    def __call__(self, image, **kwargs):
        '''
        Args:
            image (array): The image to segment as an array of shape [H,W,C]
            **kwargs: Additional arguments for the segmenter
        Returns (array, array): 
            [H,W] array of segment indices
            [S,H,W] array with masks for each of the S segments
        '''
        segments = self.segmenter(image, **self.kwargs, **kwargs)
        segment_masks = segments_to_masks(segments)
        return segments, segment_masks

class GridSegmenter():
    '''
    Segments an image into a grid with a given amount of rows and columns.
    Args:
        h_segments (int): Nr of horizontal divisions (columns) of the image
        v_segments (int): Nr of vertical divisions (rows) of the image
        bilinear (bool): Whether to fade the masks with bilinear upscaling

    '''
    def __init__(self, h_segments, v_segments, bilinear=False):
        self.h_nr = h_segments
        self.v_nr = v_segments
        self.bilinear = bilinear

    def __call__(self, image):
        '''
        Args:
            image (array): The image to segment as an array of shape [H,W,C]
        Returns (array, array): 
            [H,W] array of segment indices
            [S,H,W] array with masks for each of the S segments
        '''

        h_size, v_size = image.shape[0:2]
        segments = np.arange(self.h_nr*self.v_nr).reshape(self.h_nr, self.v_nr)
        segments = resize(
            segments, (h_size,v_size), order=0, mode='reflect',
            anti_aliasing=False
        )
        segment_masks = np.zeros((self.h_nr*self.v_nr, self.h_nr, self.v_nr))
        for i in range(self.h_nr*self.v_nr):
            segment_masks[i, i//self.h_nr, i%self.v_nr] = 1
        segment_masks = resize(
            segment_masks, (self.h_nr*self.v_nr,h_size,v_size), mode='reflect',
            order=1 if self.bilinear else 0, anti_aliasing=False
        )
        return segments, segment_masks


# Segment mask transforms
class FadeMaskTransformer():
    '''
    Fades segment masks smoothly using scipy.ndimage.gaussian_filter().
    Args:
        sigma (float/(float,float)): St.dev.(s) for the gaussian filter
        **kwargs: Additional arguments for scipy.ndimage.gaussian_filter
    Returns (array): [S,H,W] array of segment masks faded between 0 and 1
    '''
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
        self.kwargs = kwargs
    
    def __call__(self, segement_masks, **kwargs):
        '''
        Args:
            segment_masks (array): [S,H,W] array with S segment masks to fade
            **kwargs: Additional arguments for scipy.ndimage.gaussian_filter
        Returns (array): [S,H,W] array of segment masks faded between 0 and 1
    '''
        return gaussian_filter(
            segement_masks, sigma=self.sigma,
            axes=range(1,len(segement_masks.shape)), **self.kwargs, **kwargs
        )


# Segmentation utilities
def segments_to_masks(segments):
    '''
    Takes a segmented image and creates an array of masks for each segment.
    Args:
        segments (array): [H,W] array of pixels labeled by 1 of S segments
    Returns (array): [S,H,W] array with 0/1 masks for each of the S segments
    '''
    segment_ids = np.unique(segments)
    segment_masks = np.zeros(
        (segment_ids.shape[0],segments.shape[0], segments.shape[1])
        )
    for id in segment_ids:
        segment_masks[id] = np.where(segments==id,1.0,0.0)
    return segment_masks


# Perturbation mask creators
def perturbation_masks(segment_masks, samples):
    '''
    Creates images masks indicating where the image should be perturbed using
    masks indicating where each segment is and rows of samples denoting which
    segments to include in each sample.
    Args:
        segment_masks (array): [S,H,W] array with masks for each segment
        samples (array): [N,S] array with 0 indicating the segments to perturb
    Returns (array): [N,H,W] array of masks in [0,1] (lower=more perturbation)
    '''
    return np.tensordot(samples, segment_masks, axes=(1,0))