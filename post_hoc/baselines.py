import numpy as np

from .image_segmenters import (
    perturbation_masks, shift_perturbation_masks, GridSegmenter,
    WrapperSegmenter
)
from .samplers import (
    UniqueRandomSampler, SingleFeatureSampler, MultiplyWrapperSampler
)
from .image_perturbers import (
    SingleColorPerturber, SegmentColorPerturber, ColorHistogramPerturber
)
from .explainers import (
    RISEAttributer, ScikitLIMEAttributer, OriginalCIUAttributer, PDAAttributer
)
from .connectors import SegmentationAttribuitionPipeline

'''
This file contains implementations equivalent or analogous to the original or
default implementation of many perturbation-based methods.
'''

class RISEPipeline():
    '''
    An implementation of the RISE method that is mostly equivalent to the
    original implementation. By default this gives the original implementation
    but some parameters may be altered.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        pad_image (int, int): How much to extend the image beyond borders
        pad_mode (str): Which np.pad mode to use for extending the image
        batch_size (int): How many perturbed images to feed the model at once
    '''
    def __init__(
        self, segmenter=None, sampler=None, perturber=None, pad_image=(1/7,1/7),
        pad_mode='reflect', batch_size=None
    ):
        self.segmenter = segmenter
        if segmenter is None:
            self.segmenter = GridSegmenter(7, 7, True)
        self.sampler = sampler
        if sampler is None:
            self.sampler = UniqueRandomSampler()
        self.perturber = perturber
        if perturber is None:
            # This is the correct RISE perturber for ImageNet, but not otherwise
            self.perturber = SingleColorPerturber((0.485, 0.456, 0.406))
        self.explainer = RISEAttributer()
        self.pad_image = pad_image
        self.pad_mode = pad_mode
        self.batch_size = batch_size

        # Variables to hold information between calls
        self.masks = None
        self.transformed_masks = None
        self.distortion_masks = None
        self.samples = None
        self.nr_model_calls = None
        self.ys = None

    def __call__(self, image, model, sample_size=None, samples=None):
        '''
        Args:
            image (array): the [H,W,C] image to explain with RISE attributing
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of perturbed images to use to attribute
            samples (array): [N,M] alternative to sampler (replaces sample_size)
        Returdns:
            array: [M,O] output for each of the [M,H,W,C] perturbed images
            array: [M,S] samples indexing the perturbed segments of the images
        '''
        if sum(self.pad_image) > 0:
            h = int(image.shape[1]*self.pad_image[0])
            v = int(image.shape[0]*self.pad_image[1])
            pad_image = np.pad(
                image, ((0,v),(0,h),(0,0)),
                mode=self.pad_mode
            )
        else:
            h = v = 0
            pad_image = image
        segments, self.masks, self.transformed_masks = self.segmenter(pad_image)
        if not (sample_size is None or samples is None):
            raise ValueError('Both sample_size and samples cannot be set')
        if samples is None:
            self.samples = self.sampler(len(self.transformed_masks),sample_size)
        else:
            self.samples = samples

        if self.batch_size is None:
            batch = [0, len(self.samples)]
        else:
            batch = list(range(
                0, len(self.samples), self.batch_size
            )) + [len(self.samples)]
        ys = []
        perturbed_samples = []
        distortion_masks = []
        for k in range(len(batch)-1):
            distortion_mask = perturbation_masks(
                self.transformed_masks, self.samples[batch[k]:batch[k+1]]
            )
            shift_vals = np.full((len(distortion_mask),2),h)
            shift_vals[:,1] = v
            distortion_mask = shift_perturbation_masks(
                distortion_mask, shift_vals, random=True
            )
            distortion_mask = distortion_mask[
                : ,v:, h:
            ]
            perturbed_images, perturbed_sample = self.perturber(
                image, distortion_mask, self.samples[batch[k]:batch[k+1]],
                self.masks
            )
            distortion_masks.append(distortion_mask)
            perturbed_samples.append(perturbed_sample)
            ys.append(model(perturbed_images))
        perturbed_samples = np.concatenate(perturbed_samples)
        self.distortion_masks = np.concatenate(distortion_masks)
        ys = np.concatenate(ys)
        self.nr_model_calls = len(ys)
        if len(ys.shape) == 1:
            ys = np.expand_dims(self.ys, axis=-1)
        self.ys = ys
        return [self.explainer(y, self.distortion_masks) for y in ys.T]

def LIMEPipeline(
    segmenter=None, sampler=None, perturber=None, explainers=None,
    per_pixel=False, batch_size=None
):
    '''
    An implementation of LIME for images based on the default implementation in
    the lime python package. By default this gives the original implementation
    but some parameters may be altered.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainers ([callable]): Attributes features from samples and outputs
        per_pixel (bool): Whether to also return attribution maps per pixel
        batch_size (int): How many perturbed images to feed the model at once
    '''
    if segmenter is None:
        from skimage.segmentation import quickshift
        segmenter = WrapperSegmenter(
            quickshift, kernel_size=4, max_dist=200, ratio=0.2
        )
    if sampler is None:
        sampler = UniqueRandomSampler()
    if perturber is None:
        perturber = SegmentColorPerturber(mode='mean')
    if explainers is None:
        from sklearn.linear_model import Ridge
        regressor = Ridge(alpha=1, fit_intercept=True)
        from sklearn.metrics import pairwise_distances
        def kernel(Z):
            d = pairwise_distances(
                Z, np.ones((1,Z.shape[1])), metric='cosine'
            ).ravel()
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        explainers = [ScikitLIMEAttributer(regressor=regressor, kernel=kernel)]
    return SegmentationAttribuitionPipeline(
        segmenter, sampler, perturber, explainers, per_pixel, batch_size=None
    )

def CIUPipeline(
    segmenter=None, sampler=None, perturber=None, explainers=None,
    strategy='straight', per_pixel=False, batch_size=None
):
    '''
    An implementation of CIU for images based on the default implementation in
    the py_ciu_image python package. By default gives the same behavior as the
    py_ciu_image packages does, but some parameters can be changes. The strategy
    parameter can be used to decide whether CIU is calculated for each feature
    by perturbing that feature (straight) or all other features (inverse).

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainers ([callable]): Attributes features from samples and outputs
        strategy (str): Calculate effect by removing the feature or all others
        per_pixel (bool): Whether to also return attribution maps per pixel
        batch_size (int): How many perturbed images to feed the model at once
    '''
    inverse = strategy=='inverse'
    if segmenter is None:
        from skimage.segmentation import slic
        segmenter = WrapperSegmenter(
            slic, n_segments=50, compactness=10, start_label=0
        )
    if sampler is None:
        sampler = SingleFeatureSampler(inverse=inverse, add_none=True)
    if perturber is None:
        perturber = SingleColorPerturber((0.745,0.745,0.745))
    if explainers is None:
        explainers = [OriginalCIUAttributer()]
    return SegmentationAttribuitionPipeline(
        segmenter, sampler, perturber, explainers, per_pixel, batch_size=None
    )

def PDAPipeline(
    segmenter=None, sampler=None, perturber=None, explainers=None,
    mode='evidence', per_pixel=False, batch_size=None
):
    '''
    An implementation of PDA for images based on the implementation details from
    the "Explain Black-box Image Classifications Using Superpixel-based
    Interpretation" paper which introduces PDA for images. By default it gives
    the same behavior but some parameters can be changes. If no explainer is
    provided the mode parameter will determine how PDA is calculated.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainers ([callable]): Attributes features from samples and outputs
        mode (str): PDA mode to use ('probdiff', 'infodiff', 'evidence')
        per_pixel (bool): Whether to also return attribution maps per pixel
        batch_size (int): How many perturbed images to feed the model at once
    '''
    if segmenter is None:
        # Value of compactness is not specified in the paper
        from skimage.segmentation import slic
        segmenter = WrapperSegmenter(
            slic, n_segments=200, compactness=10, start_label=0
        )
    if sampler is None:
        sampler = MultiplyWrapperSampler(
            SingleFeatureSampler(add_none=True), scalar=10, multiply_none=False
        )
    if perturber is None:
        perturber = ColorHistogramPerturber(nr_bins=8)
    if explainers is None:
        explainers = [PDAAttributer(mode=mode)]
    return SegmentationAttribuitionPipeline(
        segmenter, sampler, perturber, explainers, per_pixel, batch_size=None
    )


"""
def SHAPPipeline(
    segmenter=None, sampler=None, perturber=None, explainer=None,
    per_pixel=False, batch_size=None, masker=None
):
    '''
    An implementation of SHAP for images based on the default implementation in
    the shap package. As shap does not provide one specific default version of
    the image explainer the masker parameters need to be explicitly specified
    unless an perturber is provided. The masker is either an image to replace
    the original image with or

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainer (callable): Calculates attribution based on perturbation
        per_pixel (bool): Whether to also return attribution maps per pixel
        batch_size (int): How many perturbed images to feed the model at once
        masker (array/str): blur(xsize, ysize), inpaint_telea, or inpaint_ns
    '''
    if (perturber is None) + (masker is None) == 1:
        raise ValueError('Exactly one of perturber and masker must be set')
    if segmenter is None:

    if sampler is None:

    if perturber is None:

    if explainer is None:
"""