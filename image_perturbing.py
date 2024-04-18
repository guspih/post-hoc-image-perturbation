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
def slic_segmenter(image, n_segments, compactness, **kwargs):
    '''
    Segments an image using the SLIC segmentation from skimage.
    Args:
        image (array): The image to segment as an array of shape [H,W,C]
        n_segments (int): The approx. number of segments to produce
        compactness (float): How to prioritize spatial closeness of segments
        **kwargs: Additional arguments for skimage.segmentation.slic()
    Returns (array, array): 
        [H,W] array of segment indices
        [S,H,W] array with masks for each of the S segments
    '''
    segments = slic(
        image, n_segments=n_segments,compactness=compactness, start_label=0,
        **kwargs
    )
    segment_masks = segments_to_masks(segments)
    return segments, segment_masks

def grid_segmenter(image, h_segments, v_segments, bilinear=False):
    '''
    Segments an image into a grid with a given amount of rows and columns.
    Args:
        image (array): The image to segment as an array of shape [H,W,C]
        h_segments (int): Nr of horizontal divisions (columns) of the image
        v_segments (int): Nr of vertical divisions (rows) of the image
        bilinear (bool): Whether to fade the masks with bilinear upscaling
    Returns (array, array): 
        [H,W] array of segment indices
        [S,H,W] array with masks for each of the S segments
    '''
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
        order=1 if bilinear else 0, mode='reflect', anti_aliasing=False
    )
    
    return segments, segment_masks


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

def fade_segment_masks(segement_masks, sigma, **kwargs):
    '''
    Fades segment masks smoothly using scipy.ndimage.gaussian_filter().
    Args:
        segment_masks (array): [S,H,W] array with S segment masks to fade
        sigma (float/(float,float)): St.dev.(s) for the gaussian filter
        **kwargs: Additional arguments for scipy.ndimage.gaussian_filter
    Returns (array): [S,H,W] array of segment masks faded between 0 and 1
    '''
    return gaussian_filter(
        segement_masks, sigma=sigma, axes=range(1,len(segement_masks.shape)),
        **kwargs
    )


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


# Perturbers
def single_color_perturbation(image, sample_masks, samples, color):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with a given color. Pixels to replace indicated
    by 0 and values between 0 and 1 will fade between their current color and
    the given color.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        color (float,float,float): The color to replace pixels by in RGB
    Returns (array, array):
        [N,H,W,C] array of perturbed versions of the image
        [N,S] array identical to samples
    '''
    if not issubclass(image.dtype.type, numbers.Integral):
        if isinstance(color[0], numbers.Integral):
            color = np.array(color)/255

    perturbed_segments = np.tile(image-color, (sample_masks.shape[0],1,1,1))
    perturbed_segments = (
        perturbed_segments*sample_masks.reshape(list(sample_masks.shape)+[1])
    )+color
    if isinstance(color[0], numbers.Integral):
        perturbed_segments = perturbed_segments.astype(int, copy=False)
    return perturbed_segments, samples

def replace_image_perturbation(image, sample_masks, samples, replace_images):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the corresponding pixel from other images.
    Pixels to replace indicated by 0 and values between 0 and 1 will fade
    between the color of the corresponding pixels.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        replace_images (array): [X,H,W,C] array with alternative images
    Returns (array, array):
        [N*X,H,W,C] array of perturbed versions of the image
        [N*X,S] array indicating which segments have been perturbed
    '''
    image_is_int = issubclass(image.dtype.type, numbers.Integral)
    replace_is_int = issubclass(replace_images.dtype.type, numbers.Integral)
    if image_is_int != replace_is_int:
        if image_is_int: image = image/255
        else: replace = replace/255
    if len(replace_images.shape) == 4:
        total_samples = sample_masks.shape[0]*replace_images.shape[0]
    elif len(replace_images.shape) == 3:
        total_samples = sample_masks.shape[0]
    perturbed = np.tile(image-replace_images, (total_samples,1,1,1))
    perturbed = (
        perturbed*sample_masks.reshape(list(sample_masks.shape)+[1])
    )+replace_images
    if issubclass(image.dtype.type, numbers.Integral):
        perturbed = perturbed.astype(int, copy=False)
    return perturbed, samples

def transform_perturbation(image, sample_masks, samples, transform, **kwargs):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the corresponding pixel from a transformed
    version of the image. Pixels to replace indicated by 0 and values between
    0 and 1 will fade between the color of the corresponding pixels.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        transform (callable): Callable that takes image and transforms it
        kwargs: Additional arguments for the transform
    Returns (array, array):
        [N,H,W,C] array of perturbed versions of the image
        [N,S] array identical to samples
    '''
    return replace_image_perturbation(
        image, sample_masks, samples, replace_images=transform(image,**kwargs)
    )


