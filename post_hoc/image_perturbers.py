import numpy as np

# Perturbers
class SingleColorPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each perturbation mask with a given color. Pixels to replace indicated by 0
    and values between 0 and 1 will fade between their current color and the
    given color. If color is 'mean' or 'median' the color is instead chosen as
    the mean or median of the image.

    Args:
        color (float,float,float): The color to replace pixels or mean/median
    '''
    def __init__(self, color='mean'):
        if isinstance(color, str):
            self.color = color
            if color == 'mean':
                self.get_color = lambda x: np.array(np.mean(x, axis=(0,1)))
            elif color == 'median':
                self.get_color = lambda x: np.array(np.median(x, axis=(0,1)))
            else:
                raise ValueError(
                    f'color must be RGB, "mean", or "median" but got {color}'
                )
        else:
            self.color = np.array(color)
        self.deterministic = True

    def __str__(self):
        if isinstance(self.color, str):
            color = self.color
        else:
            color = '('+','.join([str(c) for c in self.color])+')'
        return f'SingleColorPerturber({color})'

    def __call__(self, image, sample_masks, samples, segment_masks):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
        Returns:
            array: [N,H,W,C] the perturbed versions of the image
            array: [N,S] identical to samples
        '''
        color = self.color
        if isinstance(color, str):
            color = self.get_color(image)
        perturbed_segments = np.tile(image-color, (sample_masks.shape[0],1,1,1))
        perturbed_segments = (
            perturbed_segments*sample_masks.reshape(
                list(sample_masks.shape)+[1]
            )
        )+color
        return perturbed_segments, samples

class ReplaceImagePerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each perturbation mask with the corresponding pixel from other images.
    Pixels to replace indicated by 0 and values between 0 and 1 will fade
    between the color of the corresponding pixels. If replace_images is None the
    replace images are expected as part of the call.

    Args:
        replace_images (array): [X,H,W,C] array with alternative images or None
        one_each (bool): If True, each perturbed image has a given replacement
        random (bool): If True, one replace image is chosen randomly per sample
        replace_images_str (str): Used for printing perturber
    '''
    def __init__(
        self, replace_images, one_each=False, random=False,
        replace_images_str=None
    ):
        self.replace_images = replace_images
        self.one_each = one_each
        self.random = random
        self.replace_images_str = replace_images_str
        if replace_images_str is None:
            shape = 'None' if replace_images is None else replace_images.shape
            self.replace_images_str = '('+','.join([str(n) for n in shape])+')'
        self.deterministic = not random

    def __str__(self):
        content = f'{self.one_each},{self.random},{self.replace_images_str}'
        return f'ReplaceImagePerturber({content})'

    def __call__(
            self, image, sample_masks, samples, segment_masks,
            replace_images=None
        ):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
            replace_images (array): [X,H,W,C] images if None provided initially
        Returns:
            array: [N*X,H,W,C] perturbed versions of the image
            array: [N*X,S] index of which segments have been perturbed
        '''
        one_each = self.one_each
        if replace_images is None:
            replace_images = self.replace_images
        if self.random:
            one_each = True
            replace_images = replace_images[
                np.random.choice(len(replace_images), len(sample_masks))
            ]
        return replace_image_perturbation(
            image, sample_masks, samples, replace_images, one_each
        )

class TransformPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each perturbation mask with the corresponding pixel from a transformed
    version of the image. Pixels to replace indicated by 0 and values between 0
    and 1 will fade between the color of the corresponding pixels.

    Args:
        transform (callable): Callable that takes image and transforms it
        kwargs: Additional arguments for the transform
    '''
    def __init__(self, transform, **kwargs):
        self.transform = transform
        self.kwargs = kwargs
        if hasattr(transform, '__name__'):
            self.transform_str = transform.__name__
        else:
            self.transform_str = transform.__class__.__name__
        self.deterministic = False

    def __str__(self):
        kw = ','.join(np.sort([f'{k}={self.kwargs[k]}' for k in self.kwargs]))
        kw = ',' + kw if len(kw) > 0 else kw
        return f'TransformPerturber({self.transform_str}{kw})'

    def __call__(self, image, sample_masks, samples, segment_masks, **kwargs):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
            kwargs: Additional arguments for the transform
        Returns:
            array: [N,H,W,C] perturbed versions of the image
            array: [N,S] identical to samples
        '''
        replace_images = self.transform (image, **self.kwargs, **kwargs)
        return replace_image_perturbation(
            image, sample_masks, samples, replace_images
        )

class Cv2InpaintPerturber():
    '''
    Creates perturbed versions of the image by inpainting the pixels indicated
    by 0 using either of the two inpainting methods implemented by OpenCV.

    Args:
        radius (int): The radius around the masked pixels to also be inpainted
        mode (str): The inpainting method, either "telea" or "bertalmio"
    '''
    def __init__(self, mode='telea', radius=1):
        import cv2
        if mode == 'telea':
            self.flags = cv2.INPAINT_TELEA
        elif mode == 'bertalmio':
            self.flags = cv2.INPAINT_NS
        else:
            raise ValueError(
                f'mode has to be "telea" or "bertalmio", but {mode} was given.'
            )
        self.mode = mode
        self.radius = radius
        self.deterministic = True
        self.inpaint = cv2.inpaint

    def __str__(self):
        return f'Cv2InpaintPerturber({self.mode},{self.radius})'

    def __call__(self, image, sample_masks, samples, segment_masks):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
        Returns:
            array: [N,H,W,C] perturbed versions of the image
            array: [N,S] identical to samples
        '''
        image = (image*255).astype(np.uint8)
        sample_masks = 1-sample_masks.round().astype(np.uint8)
        perturbed_images = np.zeros((*sample_masks.shape,3), dtype=np.uint8)
        for i, mask in enumerate(sample_masks):
            perturbed_image = self.inpaint(image, mask, self.radius, self.flags)
            perturbed_images[i] = perturbed_image
        perturbed_image = (perturbed_image/255).astype(np.float32)
        return perturbed_images, samples

class ColorHistogramPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each perturbation mask with the median color of one bins of a histogram of
    the image chosen randomly weighted by the size of the bins. Pixels to
    replace indicated by 0 and values between 0 and 1 will fade between the
    color of the corresponding pixels.

    Args:
        nr_bins (int): The number of bins to split each color channel into
    '''
    def __init__(self, nr_bins=8):
        self.nr_bins = 8
        self.deterministic = False

    def __str__(self):
        return f'ColorHistogramPerturber({self.nr_bins})'

    def __call__(self, image, sample_masks, samples, segment_masks):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
        Returns:
            array: [N*X,H,W,C] perturbed versions of the image
            array: [N*X,S] index of which segments have been perturbed
        '''
        if np.issubdtype(image.dtype, np.integer):
            color_max = 255
        else:
            color_max = 1.0
        bin_edges = np.linspace(0, color_max, self.nr_bins+1)[1:-1]
        flat_image = image.reshape(-1,3)
        pixel_bins = np.array([
            np.digitize(channel, bin_edges) for channel in flat_image.T
        ]).T
        bins = {}
        for pixel, bin in zip(flat_image,pixel_bins):
            bin = tuple(bin)
            if not bin in bins:
                bins[bin] = [pixel]
            else:
                bins[bin].append(pixel)
        bin_medians = []
        bin_probs = []
        for bin, colors in bins.items():
            bin_medians.append(np.median(colors, axis=0))
            bin_probs.append(len(colors)/(image.shape[0]*image.shape[1]))
        bin_medians = np.array(bin_medians)
        replace_colors = bin_medians[np.random.choice(
            range(len(bin_medians)), sample_masks.shape[0], p=bin_probs
        )]
        replace_colors = np.reshape(
            replace_colors, (sample_masks.shape[0],1,1,3)
        )
        perturbed_segments = np.tile(image, (sample_masks.shape[0],1,1,1))-replace_colors
        perturbed_segments = (
            perturbed_segments*sample_masks.reshape(list(sample_masks.shape)+[1])
        )+replace_colors
        return perturbed_segments, samples

class RandomColorPerturber():
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each mask with a randomly drawn color. Pixels to replace indicated by 0 and
    values between 0 and 1 will fade between their current color and the drawn
    color. Colors are drawn by selecting a random pixel or uniformly from the
    RGB domain. The perturbed images can use the same color or one each.

    Args:
        uniform_rgb (bool): Whether to draw the random color uniformly from RGB
        draw_for_each (bool): Whether to randomize a color for each
    '''
    def __init__(self, uniform_rgb=False, draw_for_each=False):
        self.uniform_rgb = uniform_rgb
        self.draw_for_each = draw_for_each
        self.deterministic = False

    def __str__(self):
        return f'RandomColorPerturber({self.uniform_rgb},{self.draw_for_each})'

    def __call__(self, image, sample_masks, samples, segment_masks):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
        Returns:
            array: [N,H,W,C] perturbed versions of the image
            array: [N,S] identical to samples
        '''
        if self.draw_for_each:
            nr = sample_masks.shape[0]
        else:
            nr = 1
        if self.uniform_rgb:
            color = np.random.random_sample(size=(nr,1,1,3)).astype(np.float32)
        else:
            color = np.random.randint(0, image.shape[0]*image.shape[1], nr)
            color = (image.reshape(-1,3)[color]).reshape((nr,1,1,3))
        perturbed_segments = np.tile(image,(sample_masks.shape[0],1,1,1))-color
        perturbed_segments = (
            perturbed_segments*sample_masks.reshape(*sample_masks.shape,1)
        )+color
        return perturbed_segments, samples

class SegmentColorPerturber:
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each mask with a color generated from based on the sample it belongs to.
    Pixels to replace indicated by 0 and values between 0 and 1 will fade
    between the original and generated color. The color can be chosen as the
    mean or median of the segment, or drawn randomly from it.

    Args:
        mode (str): How segment color is chosen (mean, median, random)
    '''
    def __init__(self, mode='mean'):
        self.mode = mode
        self.deterministic = mode!='random'

    def __str__(self):
        return f'SegmentColorPerturber({self.mode})'

    def __call__(self, image, sample_masks, samples, segment_masks):
        '''
        Args:
            image (array): [H,W,C] array with the image to perturb
            sample_masks (array): [N,H,W] array of masks in [0,1]
            samples (array): [N,S] array indicating the perturbed segments
            segment_masks (array): [S,H,W] array with masks for each segment
        Returns:
            array: [N,H,W,C] perturbed versions of the image
            array: [N,S] identical to samples
        '''
        img = image.reshape(-1, image.shape[2])
        segment_masks = segment_masks.reshape(segment_masks.shape[0], -1)
        colors = []
        for segment_mask in segment_masks:
            if self.mode == 'mean':
                sm = segment_mask.reshape(*segment_mask.shape,1)
                colors.append(
                    np.sum(img*sm,0)/np.sum(segment_mask,0)
                )
            elif self.mode == 'median':
                color = np.zeros(3)
                for c in range(3):
                    channel = img[:,c]
                    indices = np.argsort(channel)
                    cumsum = segment_mask[indices].cumsum()
                    color[c] = channel[indices][cumsum>=(cumsum[-1]/2)][0]
                colors.append(color)
            elif self.mode == 'random':
                idx = np.random.choice(
                    img.shape[0], p=segment_mask/np.sum(segment_mask)
                )
                colors.append(img.reshape(-1,3)[idx])
        colors = np.array(colors)
        colors = colors.reshape(colors.shape[0],1,colors.shape[1])
        segment_masks = segment_masks.reshape(*segment_masks.shape,1)
        img = np.sum(segment_masks*colors, axis=0)/np.sum(segment_masks, axis=0)
        img = img.reshape(image.shape)
        return replace_image_perturbation(image, sample_masks, samples, img)


# Perturbation utilities
def replace_image_perturbation(
    image, sample_masks, samples, replace_images, one_each=False
):
    '''
    Creates perturbed versions of the image by replacing the pixels indicated by
    each perturbation mask with the corresponding pixel from other images.
    Pixels to replace indicated by 0 and values between 0 and 1 will fade
    between the color of the corresponding pixels.

    Args:
        image (array): [H,W,C] array with the image to perturb
        sample_masks (array): [N,H,W] array of masks in [0,1]
        samples (array): [N,S] array indicating the perturbed segments
        replace_images (array): [X,H,W,C] array with alternative images
        one_each (bool): If True, each perturbed image has a given replacement
    Returns:
        array: [N*X,H,W,C] perturbed versions of the image
        array: [N*X,S] index of which segments have been perturbed
    '''
    if len(replace_images.shape) == 4 and not one_each:
        total_samples = sample_masks.shape[0]*replace_images.shape[0]
    elif len(replace_images.shape) == 3 or one_each:
        total_samples = sample_masks.shape[0]
        if len(replace_images.shape) == 3:
            replace_images=replace_images.reshape(
                [1]+list(replace_images.shape)
            )
    img_tiles = round(total_samples/replace_images.shape[0])
    sample_reps = round(total_samples/sample_masks.shape[0])
    replace_images = np.tile(replace_images, (img_tiles,1,1,1))
    perturbed = image-replace_images
    sample_masks = np.repeat(sample_masks, sample_reps, axis=0)
    perturbed = (
        perturbed*sample_masks.reshape(list(sample_masks.shape)+[1])
    )+replace_images
    if np.issubdtype(image.dtype, np.integer):
        perturbed = perturbed.astype(int, copy=False)
    return perturbed, samples