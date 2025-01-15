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
        steps (int): Over how many steps to iteratively occlude the image
        normalize (bool): Whether to normalize so the curve(s) goes from 1 to 0
        return_curves (bool): Whether to also return the model outputs
        return_visuals (bool): Whether to also return the deletion images
    '''
    def __init__(
        self, mode='srg', perturber=None, steps=10, normalize=False,
        return_curves=False, return_visuals=False
    ):
        self.explanations_used = 1 # Nr of attributions used by this evaluator
        self.header = [] # Names of the scores returned by this evaluator
        self.mif = []
        if mode != 'mif':
            self.header.append('lif')
            self.mif.append(False)
        if mode != 'lif':
            self.header.append('mif')
            self.mif.append(True)
        if mode == 'srg':
            self.header.append('srg')
        if normalize:
            self.header = self.header + ['norm_'+a for a in self.header]
        self.mode = mode
        self.perturber = perturber
        self.steps = steps
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
            f'{self.mode},{self.perturber},{self.steps},{self.normalize}'
        )
        return f'ImageAUCEvaluator({content})'

    def title(self):
        return f'auc_{self.mode}' + ('_norm' if self.normalize else '')

    def __call__(
        self, image, model, vals, masks=None, model_idxs=..., **kwargs
    ):
        '''
        Args:
            image (array): [H,W,C] array of the image that has been attributed
            model (callable): Model used to predict from batches of images
            vals (array): Array of attribution scores for each feature
            masks (array): [S,H,W] array of segment masks (None=vals per pixel)
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
            samples = auc_sampler(vals, self.steps, mif)
            if not masks is None:
                distortion_masks = perturbation_masks(masks, samples)
            else:
                distortion_masks = samples
            perturbed_images, perturbed_samples = self.perturber(
                image, distortion_masks, samples, masks
            )
            visuals.append(perturbed_images)
            ys = model(perturbed_images)[:, model_idxs]
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

class LocalizationEvaluator():
    '''
    Calculates the Pointing Game score of an attribution as the fraction of the
    most attributed pixels that lie within the ground truth segmentation.
    '''
    def __init__(self):
        self.explanations_used = 1 # Nr of attributions used by this evaluator
        self.header = ['pointing_game'] # The scores returned by this evaluator

    def __call__(
        self, target_segment, vals, masks=None, model_idxs=..., **kwargs
    ):
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
        target_segment = target_segment.reshape(-1)
        return [np.mean((target_segment==model_idxs)[np.max(vals) == vals])]

    def __str__(self): return 'LocalizationEvaluator()'

    def title(self): return 'localization'


class TargetDifferenceEvaluator():
    '''
    Uses the given callable to calculate a list of similarity/distance scores
    between the input explanations. The two input explanations are assumed to
    come from different target classes.

    Args:
        metrics (callable): Takes two explanations and returns [distance]
    '''
    def __init__(self, metrics):
        self.explanations_used = 2 # Nr of attributions used by this evaluator
        self.header = metrics.header # The scores returned by this evaluator
        self.metrics = metrics

    def __str__(self):
        return f'TargetDifferenceEvaluator({self.metrics})'

    def title(self): return 'target_diff_' + '_'.join(self.header)

    def __call__(self, vals, **kwargs):
        '''
        Args:
            vals (array): [2,...] array with the two explanations
            **kwargs (dict): Used to catch arguments that are not used
        '''
        return self.metrics(vals[0][0], vals[1][0])


class InputDifferenceEvaluator():
    '''
    Args:
        metrics (callable): Takes two explanations and returns [distance]
        steps (int): Nr of steps to incrementally increase the input noise over
        noise_max (float): The maximum fraction of noise added to the image
    '''
    def __init__(self, metrics, steps=1, noise_max=0.01):
        self.explanations_used = 1 # Nr of attributions used by this evaluator
        self.metrics = metrics
        self.steps = steps
        self.noise_max = noise_max
        self.noise_levels = np.linspace(0, self.noise_max, self.steps+1)[1:]
        self.header = []
        for noise_level in self.noise_levels:
            print(noise_level)
            self.header = self.header + [
                metric+f'_{noise_level*100:.2f}%' for metric in metrics.header
            ]

        # Variables to hold information between calls
        self.explanations = []
        self.current_explained = []

    def __str__(self):
        return ( 
            f'InputDifferenceEvaluator({self.metrics},{self.steps},'
            f'{self.noise_max})'
        )

    def title(self):
        return 'input_diff_' + '_'.join(
            self.metrics.header + [f'{a*100}%' for a in self.noise_levels]
        )

    def __call__(
        self, image, model, vals, pipeline, sample_size, attribution, explainer,
        model_idxs=..., **kwargs
    ):
        is_current = all([np.all(a==b) for a, b in zip(self.current_explained, [
            image, model, pipeline, sample_size, model_idxs
        ])])
        if (
            len(self.current_explained) == 0 or not is_current
        ):
            self.current_explained = [
                image, model, pipeline, sample_size, model_idxs
            ]
            self.explanations = []

        if len(self.explanations) == 0:
            noise = np.random.rand(*image.shape)
            for noise_level in self.noise_levels:
                noise_image = (noise-image)*noise_level + image
                self.explanations.append(pipeline(
                    noise_image, model, sample_size=sample_size,
                    output_idxs=model_idxs
                ))
        ret = []
        for noise_level, (segment_maps, pixel_maps) in zip(
            self.noise_levels, self.explanations
        ):
            if attribution == 'segments':
                maps = segment_maps
            elif attribution == 'pixels':
                maps = pixel_maps
            map = maps[pipeline.explainers.index(explainer)][0]
            ret = ret + (self.metrics(vals[0], map[0]))
        return ret


class AttributionSimilarityEvaluator():
    '''
    Calculates the similarity of two attributions according to some given
    metrics. Can be used to evaluate if attributions are different when
    explaining other outputs, different models, slightly perturbed inputs, etc.
    Available metrics are "l1" and "l2" norms, "cosine" distance, "ssim", and
    any function passed that takes two inputs and returns a similarity score.
    The attributions can be normalized to give more comparable scores, in this
    case the normalization parameters are drawn from the first attribution and
    applied to both attributions. Available normalizations are "none" for no
    changes to the attribution, "unit" for brining the first attribution
    into the range of 0 to 1, and "standard" to set mean to 0 and 

    Args:
        metrics [str]: What similarity metrics to use and return
        normalize (str): How to normalize before comparison
    '''
    def __init__(self, metrics, normalize='none'):
        if len(metrics) == 0:
            raise ValueError('metrics cannot be empty')
        if not normalize in ['none', 'unit', 'standard']:
            raise ValueError(
                f'normalize has to be one of "none", "unit", or "standard" but '
                f'got {normalize}'
            )

        self.header = [] # Names of the scores
        self.metrics = []
        norm_str = '' if normalize == 'none' else '_' + normalize
        for metric in metrics:
            if not isinstance(metric, str):
                self.metrics.append(metric)
                self.header.append(metric.__name__ + norm_str)
            else:
                self.header.append(metric + norm_str)
                if metric == 'l1':
                    self.metrics.append(self.l1)
                elif metric == 'l2':
                    self.metrics.append(self.l2)
                elif metric == 'cosine':
                    self.metrics.append(self.cosine)
                elif metric == 'ssim':
                    from skimage.metrics import structural_similarity
                    def ssim(x1, x2):
                        return structural_similarity(x1, x2, data_range=(
                            max(x1.max(),x2.max()) - min(x1.min(),x2.min())
                        ))
                    self.metrics.append(ssim)
                else:
                    raise ValueError(
                        f'metrics content must be callable or one of "l1", '
                        f'"l2", "cosine", "ssim"; got {metric}'
                    )
        self.normalize = normalize

    def l1(self, x1, x2): return np.sum(np.abs(x1-x2))

    def l2(self, x1, x2): return np.sqrt(np.sum((x1-x2)**2))

    def cosine(self, x1, x2):
        x1, x2 = x1.flatten(), x2.flatten()
        return np.dot(x1,x2) / (np.linalg.norm(x1)*np.linalg.norm(x2))

    def __str__(self):
        return f'AttributionSimilarityEvaluator({self.header},{self.normalize})'

    def __call__(self, vals1, vals2):
        if self.normalize == 'unit':
            min1 = np.min(vals1)
            max1 = np.max(vals1)
            vals1 = (vals1-min1)/(max1-min1)
            vals2 = (vals2-min1)/(max1-min1)
        elif self.normalize == 'standard':
            mean1 = np.mean(vals1)
            std1 = np.std(vals1)
            vals1 = (vals1-mean1)/std1
            vals2 = (vals2-mean1)/std1
        return [fun(vals1, vals2) for fun in self.metrics]


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