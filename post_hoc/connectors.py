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

    def __call__(self, image, model, sample_size=None, samples=None):
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
                model(perturbed_images)
            )
        ys = np.concatenate(ys)
        perturbed_samples = np.concatenate(perturbed_samples)
        self.nr_model_calls = len(ys)
        if len(ys.shape)==1:
            ys = np.expand_dims(self.ys, axis=-1)
        return ys, perturbed_samples

class SegmentationAttribuitionPipeline():
    '''
    Creates an image attribution pipeline that automatically performs and
    connects segmentation, sampling, perturbing, model prediction, and 
    attribution.

    Args:
        segmenter (callable): Returns [H,W], [S,H,W] segments and S masks
        sampler (callable): Returns [N,S] N samples of segments to perturb
        perturber (callable): Returns [M,H,W,C], [M,S] perturbed images, samples
        explainer (callable): Calculates attribution based on perturbation
        per_pixel (bool): Whether to also return a map of attribution per pixel
        batch_size (int): How many perturbed images to feed the model at once
    '''
    def __init__(
        self, segmenter, sampler, perturber, explainer, per_pixel=False,
        batch_size=None
    ):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.explainer = explainer
        self.per_pixel = per_pixel
        self.batch_size = batch_size

        # Reuse SegmentationPerturbationPipeline
        self.prediction_pipeline = SegmentationPredictionPipeline(
            segmenter, sampler, perturber, batch_size=batch_size
        )

        # Variables to hold information between calls
        self.ys = None
    
    def __getattr__(self, attr):
        '''
        Gets missing attributes from wrapped pipeline.
        '''
        if attr in self.__dict__:
            return self.__dict__[attr] 
        return getattr(self.prediction_pipeline, attr)

    def __call__(self, image, model, sample_size=None, samples=None):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of samples to use to perturb
            samples (array): [N,M] alternative to sampler (replaces sample_size)
        Returns: 
            any: List of explainer outputs for each model output O
            array, optional: List of [H,W] maps of attribution per pixel
        '''
        self.ys, perturbed_samples = self.prediction_pipeline(
            image, model, sample_size, samples
        )
        ret = [self.explainer(y, perturbed_samples) for y in self.ys.T]
        if self.per_pixel:
            pixel_map = [perturbation_masks(
                    self.prediction_pipeline.transformed_masks,
                    values[-2].reshape((1,-1))
                ) for values in ret]
            return ret, pixel_map
        return ret