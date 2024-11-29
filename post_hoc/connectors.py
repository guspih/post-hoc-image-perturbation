import numpy as np

from .image_segmenters import perturbation_masks

class SegmentationPredictionPipeline():
    '''
    Creates an image perturbation pipeline that automatically performs and
    connects segmentation, sampling, perturbing, and model prediction.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        batch_size (int): How many perturbed images to feed the model at once
    '''
    def __init__(self, segmenter, sampler, perturber, batch_size=None):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.batch_size = batch_size

        # Variables to hold information between calls
        self.masks = None
        self.transformed_masks = None
        self.samples = None
        self.nr_model_calls = None

    def __call__(
        self, image, model, sample_size=None, samples=None, output_idxs=...
    ):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of perturbed images to use to attribute
            samples (array): [N,M] alternative to sampler (replaces sample_size)
        Returns:
            array: [M,O] output for each of the [M,H,W,C] perturbed images
            array: [M,S] samples indexing the perturbed segments of the images
        '''
        segments, self.masks, self.transformed_masks = self.segmenter(image)
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
        for k in range(len(batch)-1):
            distortion_masks = perturbation_masks(
                self.transformed_masks, self.samples[batch[k]:batch[k+1]]
            )
            perturbed_images, perturbed_sample = self.perturber(
                image, distortion_masks, self.samples[batch[k]:batch[k+1]],
                self.masks
            )
            perturbed_samples.append(perturbed_sample)
            ys.append(
                model(perturbed_images)[:,output_idxs]
            )
        ys = np.concatenate(ys)
        perturbed_samples = np.concatenate(perturbed_samples)
        self.nr_model_calls = len(ys)
        if len(ys.shape)==1:
            ys = np.expand_dims(ys, axis=-1)
        return ys, perturbed_samples

class SegmentationAttribuitionPipeline():
    '''
    Creates an image attribution pipeline that automatically performs and
    connects segmentation, sampling, perturbing, model prediction, and
    attribution. Returns a list that for each explainer contains a list for each
    output with each explanation type indicated by the explanan parameter.
    The 'basic' explanan returns the exact output of the explainer,
    'segment_map' returns a [H,W] attribution map where each pixel is attributed
    the value of its segment, and 'pixel_map' returns an [H,W] attribution map
    where each pixel is attributed the weighted average value of all segments
    where the weights are how much that pixel is involved in each segment.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainers ([callable]): Attributes features from samples and outputs
        explanans ([str]): What to return ("basic", "segment_map", "pixel_map")
        prune (bool): If True, catch explainer errors and remove the explainer
        batch_size (int): How many perturbed images to feed the model at once
    '''
    def __init__(
        self, segmenter, sampler, perturber, explainers, explanans=['basic'],
        prune=False, batch_size=None
    ):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.explainers = explainers
        self.explanans = explanans
        self.prune = prune
        self.batch_size = batch_size

        # Reuse SegmentationPerturbationPipeline
        self.prediction_pipeline = SegmentationPredictionPipeline(
            segmenter, sampler, perturber, batch_size=batch_size
        )

        # Variables to hold information between calls
        self.ys = None
        self.errors = [] # List of all caught explainer errors (if prune=True)

    def __getattr__(self, attr):
        '''
        Gets missing attributes from wrapped pipeline.
        '''
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.prediction_pipeline, attr)

    def __call__(
        self, image, model, sample_size=None, samples=None, output_idxs=...
    ):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of samples to use to perturb
            samples (array): [N,M] alternative to sampler (replaces sample_size)
            output_idxs (indices): Indexes of the outputs to return
        Returns:
            [[[any]]]: Explanations per explanan, per explainer, per output
        '''
        self.ys, perturbed_samples = self.prediction_pipeline(
            image, model, sample_size, samples, output_idxs
        )
        ret = {}
        if self.prune:
            ret['basic'] = []
            to_remove = []
            for i, explainer in enumerate(self.explainers):
                try: 
                    exps = [explainer(y, perturbed_samples) for y in self.ys.T]
                    ret['basic'].append(exps)
                except Exception as e:
                    self.errors.append(str(explainer)+': '+str(e))
                    to_remove.append(i)
            removed = 0
            for i in to_remove:
                del self.explainers[i-removed]
                removed += 1
            if len(self.explainers) == 0:
                errors = '\n'.join(self.errors)
                raise RuntimeError(
                    f'All explainers have been pruned raising the following'
                    f'Exceptions\n:{errors}'
                )
        else:
            ret['basic'] = [
                [explainer(y, perturbed_samples) for y in self.ys.T]
                for explainer in self.explainers
            ]
        if 'segment_map' in self.explanans:
            ret['segment_map'] = [[perturbation_masks(
                    self.prediction_pipeline.masks,
                    values[-2].reshape((1,-1))
                ) for values in explanations] for explanations in ret['basic']]
        if 'pixel_map' in self.explanans:
            ret['pixel_map'] = [[perturbation_masks(
                    self.prediction_pipeline.transformed_masks,
                    values[-2].reshape((1,-1))
                ) for values in explanations] for explanations in ret['basic']]
        return [ret[explanan] for explanan in self.explanans]