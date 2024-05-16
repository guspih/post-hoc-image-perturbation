import numpy as np
import cv2

from image_perturbers import SingleColorPerturber, ReplaceImagePerturber, cast_image
from image_segmenters import perturbation_masks


class TopVisualizer():
    '''
    Creates a numpy image showing the top attributed segments by perturbing the
    lower attributed segments. How many top segments to display can be set as
    a specific nr (k), fraction of total attribution (p), percentage of
    segments (percent), or segments with attribution above a given treshold.
    Args:
        k (int): Nr of top segments to display
        p (float): Display segments with total attribution at least p
        percent (float): The fraction of top segments to display
        treshold (float): The treshold above which to display segments
        perturber (callable): Used to perturb lower attributed segments
    '''
    def __init__(
        self, k=None, p=None, percent=None, treshold=None, perturber=None
    ):
        self.perturber = perturber
        if perturber is None:
            self.perturber = SingleColorPerturber((190,190,190))
        if (k is None)+(p is None)+(percent is None)+(treshold is None) != 3:
            raise ValueError(
                'Exactly one of k, p, percent, and treshold should be set'
            )
        self.k = k
        self.p = p
        self.percent = percent
        self.treshold = treshold
        if not k is None:
            self.mode = 'k'
        elif not p is None:
            self.mode = 'p'
        elif not percent is None:
            self.mode = 'percent'
        else:
            self.mode = 'treshold'

    def __call__(self, values, image, segment_masks, pixel_mask=None):
        '''
        Args:
            values (array): [S] array of attribution scores per segment
            image (array): Image to visualize attribution for of shape [H,W,C]
            segment_masks (array): [S,H,W] array of 0/1 masks for the segments
            pixel_mask (array): [H,W] array of attributions per pixel (unused)
        Returns (array): [H,W,C] array visualizing the attribution on the image
        '''
        nr_segments = values.shape[0]
        to_perturb = np.ones((1,nr_segments))
        if self.mode in ['percent', 'k']:
            if self.mode == 'percent':
                self.k = int(np.ceil(self.percent*nr_segments))
            top_order = values.argsort()
            to_perturb[0, top_order[:-self.k]] = 0
        elif self.mode == 'treshold':
            to_perturb[0, values>self.treshold] = 0
        else:
            values = values-np.min(values)
            values = values/np.sum(values)
            top_order = values.argsort()
            top_values = values[top_order]
            total_value = 0
            for i, val in enumerate(top_values[::-1]):
                total_value+=val
                if total_value > self.p:
                    break
            to_perturb[0, top_order[:-(i+1)]] = 0
        distortion_mask = perturbation_masks(segment_masks, to_perturb)
        perturbed_image = self.perturber(image, distortion_mask, to_perturb)[0]
        return perturbed_image[0]

class HeatmapVisualizer():
    '''
    Creates a numpy images showing the attribution of the image by creating a
    heatmap overlaid on top of the original image.
    Args:
        normalize (bool): Whether to normalize attributions in range [0,1]
        image_weight (float): How visible should the image be under the heatmap
        colormap (cv2.ColormapTypes): The OpenCV colormap that the heatmap uses
        invert_colormap (bool): Whether to swap the direction of the colormap
    '''
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
        '''
        Args:
            values (array): [S] array of attribution scores per segment
            image (array): Image to visualize attribution for of shape [H,W,C]
            segment_masks (array): [S,H,W] array of 0/1 masks for the segments
            pixel_mask (array): [H,W] array of attributions per pixel (unused)
        Returns (array): [H,W,C] array visualizing the attribution on the image
        '''
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