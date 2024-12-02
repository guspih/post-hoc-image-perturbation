import numpy as np

from .image_perturbers import SingleColorPerturber
from .image_segmenters import perturbation_masks

#Evaluators
class ImageAUCEvaluator():
    '''
    Evaluates an image attribution through calculating the area under the
    deletion curve by deleting pixels/segments from most to least attributed
    (mif) or deleting the least attributed first (lif). Can also calculate both
    and return the difference between them (srg).

    Args:
        mode (str): Which score to calculate. 'mif'/'lif' are included in 'srg'
        perturber (callable): Perturber used to remove the segments
        sample_size (int): How many perturbed images to generate
        normalize (bool): Whether to normalize so the curve(s) goes from 1 to 0
        return_curves (bool): Whether to also return the model outputs
        return_visuals (bool): Whether to also return the deletion images
    '''
    def __init__(
        self, mode='srg', perturber=None, sample_size=10, normalize=False,
        return_curves=False, return_visuals=False
    ):
        self.mif = []
        self.header = []
        if mode != 'mif':
            self.mif.append(False)
            self.header.append('lif')
        if mode != 'lif':
            self.mif.append(True)
            self.header.append('mif')
        if mode == 'srg':
            self.header.append('srg')
        if normalize:
            self.header = self.header + ['norm_'+a for a in self.header]
        self.mode = mode
        self.perturber = perturber
        self.sample_size = sample_size
        self.normalize = normalize
        self.return_curves = return_curves
        self.return_visuals = return_visuals
        if perturber is None:
            self.perturber = SingleColorPerturber((0.5,0.5,0.5))

        # Variables to hold information between calls
        self.scores = None
        self.curves = None

    def __str__(self):
        content = (
            f'{self.mode},{self.perturber},{self.sample_size},{self.normalize}'
        )
        return f'ImageAUCEvaluator({content})'

    def title(self):
        return f'auc_{self.mode}' + ('_norm' if self.normalize else '')

    def __call__(
        self, image, model, vals, masks=None, sample_size=None, model_idxs=...,
        **kwargs
    ):
        '''
        Args:
            image (array): [H,W,C] array of the image that has been attributed
            model (callable): Model used to predict from batches of images
            vals (array): Array of attribution scores for each feature
            masks (array): [S,H,W] array of segment masks (None=vals per pixel)
            sample_size (int): Nr of deletion steps (overrides self.sample_size)
            model_idxs (index): Index for the model outputs to use
            **kwargs (dict): Used to catch arguments that are not used
        Returns:
            [array]: List of [O] AUC arrays for O model outputs for each score
            [array], optional: List of [sample_size, O] model output arrays
        '''
        scores = []
        curves = []
        visuals = []
        self.scores = []
        self.curves = []
        for mif in self.mif:
            if sample_size == None:
                sample_size = self.sample_size
            samples = auc_sampler(vals, sample_size, mif)
            if not masks is None:
                distortion_masks = perturbation_masks(masks, samples)
            else:
                distortion_masks = samples
            perturbed_images, perturbed_samples = self.perturber(
                image, distortion_masks, samples, masks
            )
            visuals.append(perturbed_images)
            ys = model(perturbed_images)[model_idxs]
            #if len(ys.shape) == 1:
            #    ys = np.expand_dims(ys, axis=-1)
            self.curves.append(ys)
            self.scores.append(np.mean(ys, axis=0))
        scores, curves = self.scores, self.curves
        if self.mode == 'srg':
            scores.append(scores[0]-scores[1])
        if self.normalize:
            norm_scores, norm_curves = self.get_normalized()
            scores = scores + norm_scores
            curves = curves + norm_curves

        ret = [scores]
        if self.return_curves:
            ret.append(curves)
        if self.return_visuals:
            ret.append(visuals)
        return ret[0] if len(ret) == 1 else ret

    def get_normalized(self):
        curves = []
        scores = []
        for ys in self.curves:
            ys = ys-ys[-1]
            ys = ys/ys[0]
            curves.append(ys)
            scores.append(np.mean(ys, axis=0))
        if self.mode == 'srg':
            scores.append(scores[0]-scores[1])
        return scores, curves

class PointingGameEvaluator():
    '''
    Calculates the Pointing Game score of an attribution as the fraction of the
    most attributed pixels that lie within the ground truth segmentation.
    '''
    def __call__(self, hit_mask, vals, masks=None):
        '''
        Args:
            hit_mask (array): [H,W] array of true segmentations of the image
            vals (array): Array of attribution scores for each image feature
            masks (array): [S,H,W] array of segment masks (None=vals per pixel)
        Returns:
            float: The Pointing Game score of the attribution
        '''
        if not masks is None:
            vals = perturbation_masks(masks, vals)
        if len(vals.shape) == 3:
            vals = np.squeeze(vals, axis=0)
        vals = vals.reshape(-1)
        hit_mask = hit_mask.reshape(-1)
        return np.mean(hit_mask[np.max(vals) == vals])

class AttributionSimilarityEvaluator():
    '''
    Calculates the similarity of two attributions according to some given
    metrics. Can be used to evaluate if attributions are different when
    explaining other outputs, different models, slightly perturbed inputs, etc.
    '''


# Evaluation samplers
def auc_sampler(vals, sample_size=None, mif=False, ignore_ends=False):
    '''
    Creates an array of samples where each following sample has more deleted
    features than the preceeding one. The most or least influential features, as
    determined by a given attribution array, are deleted first. Deleting the
    least influential first (lif) is equivalent to inserting the most
    influential first (insertion score).

    Args:
        vals (array): The attribution scores of the different features
        sample_size (int): Nr of samples to generate (<=len(vals))
        mif (bool): If True, the most influential feature is deleted first
        ignore_ends (bool): If True, does not generate samples of all 0 or 1
    Returns:
        array: [sample_size, *vals.shape] array indexing what to perturb
    '''
    if sample_size is None:
        sample_size = min(vals.size+1, 100)
    elif sample_size > vals.size+1:
        raise ValueError(
            f'sample_size must be <= vals.size+1, but '
            f'sample_size={sample_size} and vals.size={vals.size}.'
        )
    if len(vals.shape) == 3:
        vals = np.squeeze(vals, axis=0)
    shape = vals.shape
    vals = vals.reshape(-1)
    indices = np.argsort(vals)
    if mif:
        indices = indices[::-1]
    samples = np.ones((sample_size, vals.size))
    old_j=0
    if ignore_ends:
        breakpoints = np.linspace(0, vals.size, sample_size+2)[1:-1]
    else:
        breakpoints = np.linspace(0, vals.size, sample_size)
    for n, i in enumerate(breakpoints):
        j = round(i)
        idxs = indices[old_j:j]
        samples[n:, idxs] = 0
        old_j = j
    return samples.reshape(sample_size, *shape)