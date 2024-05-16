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


# Perturbers
class SingleColorPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with a given color. Pixels to replace indicated
    by 0 and values between 0 and 1 will fade between their current color and
    the given color. If color is None, then average image color is used.
    Args:
        color (float,float,float): The color to replace pixels by in RGB
    '''
    def __init__(self, color=None):
        self.color = color

    def __call__(self, image, sample_masks, samples):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
        Returns (array, array):
            [N,H,W,C] array of perturbed versions of the image
            [N,S] array identical to samples
        '''
        color = self.color
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

class ReplaceImagePerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the corresponding pixel from other images.
    Pixels to replace indicated by 0 and values between 0 and 1 will fade
    between the color of the corresponding pixels. If replace_images is None
    the replace images are expected as part of the call.
    Args:
        replace_images (array): [X,H,W,C] array with alternative images or None
        one_each (bool): If True, each perturbed image has a given replacement
    '''
    def __init__(self, replace_images, one_each=False):
        self.replace_images = replace_images
        self.one_each = one_each

    def __call__(self, image, sample_masks, samples, replace_images=None):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            replace_images (array): [X,H,W,C] images if None provided initially
        Returns (array, array):
            [N*X,H,W,C] array of perturbed versions of the image
            [N*X,S] array indicating which segments have been perturbed
        '''
        if self.replace_images is None:
            return replace_image_perturbation(
                image, sample_masks, samples, replace_images, self.one_each
            )
        return replace_image_perturbation(
            image, sample_masks, samples, self.replace_images, self.one_each
        )

class TransformPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the corresponding pixel from a transformed
    version of the image. Pixels to replace indicated by 0 and values between
    0 and 1 will fade between the color of the corresponding pixels.
    Args:
        transform (callable): Callable that takes image and transforms it
        kwargs: Additional arguments for the transform
    '''
    def __init__(self, transform, **kwargs):
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, image, sample_masks, samples, **kwargs):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            kwargs: Additional arguments for the transform
        Returns (array, array):
            [N,H,W,C] array of perturbed versions of the image
            [N,S] array identical to samples
        '''
        replace_images = self.transform (image, **self.kwargs, **kwargs)
        return replace_image_perturbation(
            image, sample_masks, samples, replace_images
        )

class Cv2InpaintPertuber():
    '''
    Creates perturbed versions of the image by inpainting the pixels indicated
    by 0 using either of the two inpainting methods implemented by OpenCV.
    Args:
        radius (int): The radius around the masked pixels to also be inpainted
        mode (str): The inpainting method, either "telea" or "bertalmio"
    '''
    def __init__(self, mode='telea', radius=1):
        if mode == 'telea':
            self.flags = cv2.INPAINT_TELEA
        elif mode == 'bertalmio':
            self.flags = cv2.INPAINT_NS
        else:
            raise ValueError(
                f'mode has to be "telea" or "bertalmio", but {mode} was given.'
            )
        self.radius = radius

    def __call__(self, image, sample_masks, samples):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
        Returns (array, array):
            [N,H,W,C] array of perturbed versions of the image
            [N,S] array identical to samples
        '''
        dtype = image.dtype.type
        image = cast_image(image, np.uint8)
        sample_masks = 1-sample_masks.round().astype(np.uint8)
        perturbed_images = np.zeros(
            list(sample_masks.shape)+[3], dtype=np.uint8
            )
        for i, mask in enumerate(sample_masks):
            perturbed_image = cv2.inpaint(image, mask, self.radius, self.flags)
            perturbed_images[i] = perturbed_image
        perturbed_image = cast_image(perturbed_image, dtype)
        return perturbed_images, samples

class ColorHistogramPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated
    by each perturbation mask with the median color of one bins of a histogram
    of the image chosen randomly weighted by the size of the bins. Pixels to
    replace indicated by 0 and values between 0 and 1 will fade between the
    color of the corresponding pixels.
    Args:
        nr_bins (int): The number of bins to split each color channel into
    '''
    def __init__(self, nr_bins=8):
        self.nr_bins = 8
    
    def __call__(self, image, sample_masks, samples):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
        Returns (array, array):
            [N*X,H,W,C] array of perturbed versions of the image
            [N*X,S] array indicating which segments have been perturbed
        '''
        if issubclass(image.dtype.type, numbers.Integral):
            color_max = 255
        else:
            color_max = 1.0
        bin_edges = np.linspace(0, color_max, self.nr_bins+1)[1:-1]
        flat_image = image.reshape(-1,3)
        pixel_bins = np.array([
            np.digitize(channel, bin_edges) for channel in flat_image.T
        ]).T
        bins = {}
        for pixel, bin in zip(flat_image,pixel_bins):
            bin = tuple(bin)
            if not bin in bins:
                bins[bin] = [pixel]
            else:
                bins[bin].append(pixel)
        bin_medians = []
        bin_probs = []
        for bin, colors in bins.items():
            bin_medians.append(np.median(colors, axis=0))
            bin_probs.append(len(colors)/(image.shape[0]*image.shape[1]))
        bin_medians = np.array(bin_medians)
        replace_colors = bin_medians[np.random.choice(
            range(len(bin_medians)), sample_masks.shape[0], p=bin_probs
        )]
        replace_colors = np.reshape(
            replace_colors, (sample_masks.shape[0],1,1,3)
        )
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


# Perturbation utilities
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