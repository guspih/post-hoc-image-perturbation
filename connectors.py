import numpy as np

from image_segmenters import perturbation_masks


class SegmentationAttribuitionPipeline():
    '''
    Creates an image attribution pipeline that automatically performs and
    connects segmentation, sampling, perturbing, model prediction, and 
    attribution.
    Args:
        segmenter (callable): Return [H,W], [S,H,W] segments and S masks
        sampler (callable): Return [N,S] N samples of segments to perturb
        perturber (callable): Return [N,H,W,C], [N,S] perturbed images, samples
        explainer (callable): Calculates attribution based on perturbation
        mask_transform (callable): Return [S,H,W] transformed masks
        per_pixel (bool): Whether to return [O,H,W] the attribution per pixel
    '''
    def __init__(
        self, segmenter, sampler, perturber, explainer, mask_transform=None,
        per_pixel=False
    ):
        self.segmenter = segmenter
        self.sampler = sampler
        self.perturber = perturber
        self.explainer = explainer
        self.mask_transform = mask_transform
        self.per_pixel = per_pixel

        # Variables to hold information between calls
        self.masks = None
        self.transformed_masks = None
        self.samples = None
        self.ys = None
    
    def __call__(self, image, model, sample_size=None):
        '''
        Args:
            image (array): the [H,W,C] image to explain via attribution
            model (callable): The prediction model returning [N,O] for output O
            sample_size (int): The nr N of perturbed images to use to attribute
        Returns: Output of explainer for each output O, or [O,H,W] if per_pixel
        '''
        segments, self.masks = self.segmenter(image)
        if not self.mask_transform is None:
            self.transformed_masks = self.mask_transform(self.masks)
        else:
            self.transformed_masks = self.masks
        self.samples = self.sampler(self.masks.shape[0], sample_size)
        distortion_masks = perturbation_masks(
            self.transformed_masks, self.samples
        )
        perturbed_images, perturbed_samples = self.perturber(
            image, distortion_masks, self.samples
        )
        self.ys = model(perturbed_images)
        if len(self.ys.shape)==1:
            self.ys = np.expand_dims(self.ys, axis=-1)
        ret = [self.explainer(y, perturbed_samples) for y in self.ys.T]
        if self.per_pixel:
            ret = [perturbation_masks(
                    values[-2], self.transformed_masks
                ) for values in ret]
        return ret