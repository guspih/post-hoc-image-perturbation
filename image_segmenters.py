import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter


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
        if hasattr(segmenter, '__name__'):
            self.segmenter_str = segmenter.__name__
        else:
            self.segmenter_str = segmenter.__class__.__name__

    def __str__(self):
        kw = ','.join(np.sort([f'{k}={self.kwargs[k]}' for k in self.kwargs]))
        kw = ',' + kw if len(kw) > 0 else kw
        return f'WrapperSegmenter({self.segmenter_str}{kw})'
    
    def __call__(self, image, **kwargs):
        '''
        Args:
            image (array): The image to segment as an array of shape [H,W,C]
            **kwargs: Additional arguments for the segmenter
        Returns (array, array): 
            [H,W] array of segment indices
            [S,H,W] array with masks for each of the S segments
            [S,H,W] array with masks for each of the S segments
        '''
        segments = self.segmenter(image, **self.kwargs, **kwargs)
        segment_masks = segments_to_masks(segments)
        return segments, segment_masks, segment_masks

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

    def __str__(self):
        return f'GridSegmenter({self.h_nr},{self.v_nr},{self.bilinear})'

    def __call__(self, image):
        '''
        Args:
            image (array): The image to segment as an array of shape [H,W,C]
        Returns (array, array): 
            [H,W] array of segment indices
            [S,H,W] array with masks for each of the S segments
            [S,H,W] array with masks for each of the S segments
        '''
        h_size, v_size = image.shape[0:2]
        segments = np.arange(self.h_nr*self.v_nr).reshape(self.h_nr, self.v_nr)
        segments = resize(
            segments, (h_size,v_size), order=0, mode='reflect',
            anti_aliasing=False
        )
        masks = np.zeros((self.h_nr*self.v_nr, self.h_nr, self.v_nr))
        for i in range(self.h_nr*self.v_nr):
            masks[i, i//self.h_nr, i%self.v_nr] = 1
        segment_masks = resize(
            masks, (self.h_nr*self.v_nr,h_size,v_size), mode='reflect',
            order=0, anti_aliasing=False
        )
        if not self.bilinear:
            return segments, segment_masks, segment_masks
        transformed_masks = resize(
            masks, (self.h_nr*self.v_nr,h_size,v_size), mode='reflect',
            order=1, anti_aliasing=False
        )   
        return segments, segment_masks, transformed_masks

class FadeMaskSegmenter():
    '''
    Segments an image using the wrapped segmenter and then smoothens the edges
    of the segmentation masks by applying a gaussian filter. The faded masks
    have values between 0 and 1 indicating the strength of pertubation to use.
    Args:
        segmenter (callable): Segments the image into segment masks
        sigma (float/(float,float)): St.dev.(s) for the gaussian filter
        **kwargs: Additional arguments for scipy.ndimage.gaussian_filter
    '''
    def __init__(self, segmenter, sigma, **kwargs):
        self.segmenter = segmenter
        self.sigma = sigma
        self.kwargs = kwargs
    
    def __str__(self):
        kw = ','.join(np.sort([f'{k}={self.kwargs[k]}' for k in self.kwargs]))
        kw = ',' + kw if len(kw) > 0 else kw
        return f'FadeMaskSegmenter({self.segmenter},{self.sigma}{kw})'

    def __call__(self, image, **kwargs):
        '''
        Args:
            image (array): The image to segment as an array of shape [H,W,C]
        Returns (array, array): 
            [H,W] array of segment indices
            [S,H,W] array with masks for each of the S segments
            [S,H,W] array with masks for each of the S segments
        '''
        segments, segment_masks, transformed_masks = self.segmenter(image)
        transformed_masks = gaussian_filter(
            transformed_masks, sigma=self.sigma,
            axes=range(1,len(transformed_masks.shape)), **self.kwargs, **kwargs
        )
        return segments, segment_masks, transformed_masks


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
    distortion_masks = np.tensordot(samples, segment_masks, axes=(1,0))
    return distortion_masks/np.sum(segment_masks, axis=0)