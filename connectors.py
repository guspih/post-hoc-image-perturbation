import numpy as np

from image_segmenters import perturbation_masks

class SegmentationPredictionPipeline():
    '''
    Creates an image perturbation pipeline that automatically performs and
    connects segmentation, sampling, perturbing, and model prediction.
    Args:
        segmenter (callable): Return [H,W], [S,H,W] segments and S masks
        sampler (callable): Return [N,S] N samples of segments to perturb
        perturber (callable): Return [M,H,W,C], [M,S] perturbed images, samples
    '''
    def __init__(self, segmenter, sampler, perturber):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber

        # Variables to hold information between calls
        self.masks = None
        self.transformed_masks = None
        self.samples = None
        self.nr_model_calls = None

    def __call__(self, image, model, sample_size=None):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of perturbed images to use to attribute
        Returns (array, array):
            [M,O] model output for each of the [M,H,W,C] perturbed images
            [M,S] samples indicating the perturbed segments in each image
        '''
        segments, self.masks, self.transformed_masks = self.segmenter(image)
        self.samples = self.sampler(len(self.transformed_masks), sample_size)
        distortion_masks = perturbation_masks(
            self.transformed_masks, self.samples
        )
        perturbed_images, perturbed_samples = self.perturber(
            image, distortion_masks, self.samples
        )
        self.nr_model_calls = len(perturbed_images)
        ys = model(perturbed_images)
        if len(ys.shape)==1:
            ys = np.expand_dims(self.ys, axis=-1)
        return ys, perturbed_samples

class SegmentationAttribuitionPipeline():
    '''
    Creates an image attribution pipeline that automatically performs and
    connects segmentation, sampling, perturbing, model prediction, and 
    attribution.
    Args:
        segmenter (callable): Return [H,W], [S,H,W] segments and S masks
        sampler (callable): Return [N,S] N samples of segments to perturb
        perturber (callable): Return [M,H,W,C], [M,S] perturbed images, samples
        explainer (callable): Calculates attribution based on perturbation
        per_pixel (bool): Whether to also return a map of attribution per pixel
    '''
    def __init__(
        self, segmenter, sampler, perturber, explainer, per_pixel=False
    ):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.explainer = explainer
        self.per_pixel = per_pixel

        # Reuse SegmentationPerturbationPipeline
        self.perturbation_pipeline = SegmentationPredictionPipeline(
            segmenter, sampler, perturber
        )

        # Variables to hold information between calls
        self.ys = None
    
    def __getattr__(self, attr):
        '''
        Gets missing attributes from wrapped pipeline.
        '''
        if attr in self.__dict__:
            return self.__dict__[attr] 
        return getattr(self.perturbation_pipeline, attr)

    def __call__(self, image, model, sample_size=None):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [M,O] for output O
            sample_size (int): The nr N of samples to use to perturb
        Returns (Any, (array, optional)): 
            List of explainer outputs for each model output O
            List of [H,W] maps of attribution per pixel (if self.per_pixel)
        '''
        self.ys, perturbed_samples = self.perturbation_pipeline(
            image, model, sample_size
        )
        ret = [self.explainer(y, perturbed_samples) for y in self.ys.T]
        if self.per_pixel:
            pixel_map = [perturbation_masks(
                    self.perturbation_pipeline.transformed_masks,
                    values[-2].reshape((1,-1))
                ) for values in ret]
            return ret, pixel_map
        return ret