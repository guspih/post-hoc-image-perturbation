import numpy as np

from image_perturbers import SingleColorPerturber
from image_segmenters import perturbation_masks

#Evaluators
class ImageAUCEvaluator():
    '''
    Evaluates an image segment attribution through calculating the area under
    the insertion/deletion curve by inserting/deleting segments starting with
    the most attributed and ending with the last. Can also calculate the SRG
    score which is the difference between the insertion and deletion scores.
    Returns the area for the given mode or insertion, deletion, and SRG area
    if mode is SRG.
    Args:
        mode (str): Which score to calculate (insertion, deletion, srg)
        perturber (callable): Perturber used to remove the segments
        return_curves (bool): Whether to also return the model outputs
    '''
    def __init__(self, mode='srg', perturber=None, return_curves=False):
        self.delete = []
        if mode != 'deletion':
            self.delete.append(False)
        if mode != 'insertion':
            self.delete.append(True)
        self.mode = mode
        self.perturber = perturber
        self.return_curves = return_curves
        if perturber is None:
            self.perturber = SingleColorPerturber((190,190,190))

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
        Returns (array, (array,optional)):
            [O,P] array of AUC for model outputs O and J=3 if mode==srg else 1
        '''
        scores = []
        curves = []
        for delete in self.delete:
            samples = auc_sampler(vals, sample_size, delete)
            if not masks is None:
                distortion_masks = perturbation_masks(masks, samples)
            else:
                distortion_masks = samples
            perturbed_images, perturbed_samples = self.perturber(
                image, distortion_masks, samples
            )
            ys = model(perturbed_images)
            if len(ys.shape)==1:
                ys = np.expand_dims(ys, axis=-1)
            curves.append(ys[model_idxs])
            scores.append(np.mean(ys, axis=0)[model_idxs])
        if self.mode == 'srg':
            scores.append(scores[0]-scores[1])
        if masks is None:
            return scores, curves, perturbed_images
        if self.return_curves:
            return scores, curves
        return scores

#class PointingGameEvaluator():
#    def __init__():
#        pass
#
#    def __call__():


# Evaluation samplers
def auc_sampler(vals, sample_size=None, deletion=False):
    '''
    Creates an array of samples where each following sample has more inserted
    or deleted samples than the preceeding one. The most influential features,
    as determined by a given attribution array, is inserted or deleted first.
    Args:
        vals (array): The attribution scores of the different features
        sample_size (int): Nr of samples to generate (<=len(vals))
        deletion (bool): If True, samples are steadily deleted, else inserted
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
    indices = np.argsort(vals)[::-1]
    if deletion:
        point = 0
        samples = np.ones((sample_size, vals.size))
    else:
        point = 1
        samples = np.zeros((sample_size, vals.size))
    old_j=0
    for n, i in enumerate(np.linspace(0,vals.size, sample_size)):
        j = round(i)
        idxs = indices[old_j:j]
        samples[n:, idxs] = point
        old_j = j
    return samples.reshape(sample_size, *shape)