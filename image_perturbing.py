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
def single_color_perturbation(image, sample_masks, samples, color=None):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with a given color. Pixels to replace indicated
    by 0 and values between 0 and 1 will fade between their current color and
    the given color. If color is None, then average image color is used.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        color (float,float,float): The color to replace pixels by in RGB
    Returns (array, array):
        [N,H,W,C] array of perturbed versions of the image
        [N,S] array identical to samples
    '''
    if color is None:
        color = np.mean(color, axis=(0,1))

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

def replace_image_perturbation(
    image, sample_masks, samples, replace_images, one_each=False
):
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
        one_each (bool): If True, each perturbed image has a given replacement
    Returns (array, array):
        [N*X,H,W,C] array of perturbed versions of the image
        [N*X,S] array indicating which segments have been perturbed
    '''
    replace_images = cast_image(replace_images, image.dtype.type)
    if len(replace_images.shape) == 4 and not one_each:
        total_samples = sample_masks.shape[0]*replace_images.shape[0]
    elif len(replace_images.shape) == 3 or one_each:
        total_samples = sample_masks.shape[0]
        if len(replace_images.shape) == 3:
            replace_images=replace_images.reshape(
                [1]+list(replace_images.shape)
            )
    img_tiles = round(total_samples/replace_images.shape[0])
    sample_reps = round(total_samples/sample_masks.shape[0])
    replace_images = np.tile(replace_images, (img_tiles,1,1,1))
    perturbed = image-replace_images
    sample_masks = np.repeat(sample_masks, sample_reps, axis=0)
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

def cv2_inpaint_perturbation(
    image, sample_masks, samples, radius=1, method='telea'
):
    '''
    Creates perturbed versions of the image by inpainting the pixels indicated
    by 0 using either of the two inpainting methods implemented by OpenCV.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        radius (int): The radius around the masked pixels to also be inpainted
        method (str): The inpainting method, either "telea" or "bertalmio"
    Returns (array, array):
        [N,H,W,C] array of perturbed versions of the image
        [N,S] array identical to samples
    '''
    dtype = image.dtype.type
    image = cast_image(image, np.uint8)
    sample_masks = 1-sample_masks.round().astype(np.uint8)
    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    elif method == 'bertalmio':
        flags = cv2.INPAINT_NS
    else:
        raise ValueError(
            f'method has to be "telea" or "bertalmio", but {method} was given.'
        )
    perturbed_images = np.zeros(list(sample_masks.shape)+[3], dtype=np.uint8)
    for i, mask in enumerate(sample_masks):
        perturbed_image = cv2.inpaint(image, mask, radius, flags)
        perturbed_images[i] = perturbed_image
    perturbed_image = cast_image(perturbed_image, dtype)
    return perturbed_images, samples

def color_histogram_perturbation(image, sample_masks, samples, nr_bins=8):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the median color of one bins of a histogram
    of the image chosen randomly weighted by the size of the bins. Pixels to
    replace indicated by 0 and values between 0 and 1 will fade between the
    color of the corresponding pixels.
    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        nr_bins (int): The number of bins to split each color channel into
    Returns (array, array):
        [N*X,H,W,C] array of perturbed versions of the image
        [N*X,S] array indicating which segments have been perturbed
    '''
    if issubclass(image.dtype.type, numbers.Integral):
        color_max = 255
    else:
        color_max = 1.0
    bin_edges = np.linspace(0, color_max, nr_bins+1)[1:-1]
    flat_image = image.reshape(-1,3)
    pixel_bins = np.array([
        np.digitize(channel, bin_edges) for channel in flat_image.T
    ]).T
    bins = {}
    for pixel, bin in zip(flat_image,pixel_bins):
        bin = tuple(bin)
        if not bin in bins:
            bins[bin] = pixel.reshape((1,3))
        else:
            bins[bin] = np.append(bins[bin], pixel.reshape((1,3)), axis=0)
    bin_medians = []
    bin_probs = []
    for bin, colors in bins.items():
        bin_medians.append(np.median(colors, axis=0))
        bin_probs.append(colors.shape[0]/(image.shape[0]*image.shape[1]))
    bin_medians = np.array(bin_medians)
    replace_colors = bin_medians[np.random.choice(
        range(len(bin_medians)), sample_masks.shape[0], p=bin_probs
    )]
    replace_colors = np.reshape(replace_colors, (sample_masks.shape[0],1,1,3))
    print(((0,0),(0,image.shape[0]-1),(0,image.shape[1]-1),(0,0)))
    replace_colors = np.pad(
        replace_colors,
        ((0,0),(0,image.shape[0]-1),(0,image.shape[1]-1),(0,0)), 'edge'
    )
    return replace_image_perturbation(
        image, sample_masks, samples, replace_colors, one_each=True
    )




# Image handling utilities
def cast_image(image, dtype):
    '''
    Casts an image from one numpy dtype to another. Integer dtypes has RGB 
    values from 0 to 255 and Fractional dtypes has values from 0.0 to 1.0
    Args:
        image (array): [H,W,C] array with the image to cast
        dtype (dtype): The numpy datatype to cast the image to
    Returns (array): [H,W,C] array with the image in the given dtype format
    '''
    image_is_int = issubclass(image.dtype.type, numbers.Integral)
    image_max_val = np.max(image)
    dtype_is_int = issubclass(dtype, numbers.Integral)
    if dtype_is_int == (image_is_int or image_max_val>1.0):
        return image.astype(dtype)
    if dtype_is_int:
        return (image*255).round().astype(dtype) 
    return (image/255).astype(dtype)
