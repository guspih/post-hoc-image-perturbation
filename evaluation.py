import numpy as np

from image_perturbers import SingleColorPerturber
from image_segmenters import perturbation_masks

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
        return_curves (bool): Whether to also return the model outputs
        normalize (bool): Whether to normalize so the curve goes from 1 to 0
    '''
    def __init__(
        self, mode='srg', perturber=None, return_curves=False, normalize=False
    ):
        self.mif = []
        if mode != 'mif':
            self.mif.append(False)
        if mode != 'lif':
            self.mif.append(True)
        self.mode = mode
        self.perturber = perturber
        self.return_curves = return_curves
        self.normalize = normalize
        if perturber is None:
            self.perturber = SingleColorPerturber((0.5,0.5,0.5))

    def __call__(
        self, image, model, vals, masks=None, sample_size=None, model_idxs=...
    ):
        '''
        Args:
            image (array): [H,W,C] array of the image that has been attributed
            model (callable): Model used to predict from batches of images
            vals (array): Array of attribution scores for each feature
            masks (array): [S,H,W] array of segment masks (None=vals per pixel)
            sample_size (int): How many perturbed images to generate
            model_idxs (index): Index for the model outputs to use
        Returns ([array], ([array],optional)):
            List of [O] arrays of AUC for O model outputs for each score
            List of [sample_size, O] arrays with all model outputs
        '''
        scores = []
        curves = []
        for mif in self.mif:
            samples = auc_sampler(vals, sample_size, mif)
            if not masks is None:
                distortion_masks = perturbation_masks(masks, samples)
            else:
                distortion_masks = samples
            perturbed_images, perturbed_samples = self.perturber(
                image, distortion_masks, samples
            )
            ys = model(perturbed_images)[model_idxs]
            if len(ys.shape) == 1:
                ys = np.expand_dims(ys, axis=-1)
            if self.normalize:
                ys = ys-ys[-1]
                ys = ys/ys[0]
            curves.append(ys)
            scores.append(np.mean(ys, axis=0))
        if self.mode == 'srg':
            scores.append(scores[0]-scores[1])
        if self.return_curves:
            return scores, curves
        return scores

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
        Returns (float): The Pointing Game score of the attribution
        '''
        if not masks is None:
            vals = perturbation_masks(masks, vals)
        if len(vals.shape) == 3:
            vals = np.squeeze(vals, axis=0)
        vals = vals.reshape(-1)
        hit_mask = hit_mask.reshape(-1)
        return np.mean(hit_mask[np.max(vals) == vals])

# Evaluation samplers
def auc_sampler(vals, sample_size=None, mif=False, ignore_ends=False):
    '''
    Creates an array of samples where each following sample has more deleted
    features than the preceeding one. The most or least influential features,
    as determined by a given attribution array, are deleted first. Deleting
    the least influential fisrst (lif) is equivalent to inserting the most 
    influential first (insertion score).
    Args:
        vals (array): The attribution scores of the different features
        sample_size (int): Nr of samples to generate (<=len(vals))
        mif (bool): If True, the most influential feature is deleted first
        ignore_ends (bool): If True, does not generate samples of all 0 or 1
    Returns (array): [sample_size, *vals.shape] array indexing what to perturb
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