import numpy as np

from .image_perturbers import SingleColorPerturber, ReplaceImagePerturber
from .image_segmenters import perturbation_masks

class TopVisualizer():
    '''
    Creates a numpy image showing the top attributed segments by perturbing the
    lower attributed segments. How many top segments to display can be set as a
    specific nr (k), fraction of total attribution (p), percentage of segments
    (percent), or segments with attribution above a given treshold.

    Args:
        k (int): Nr of top segments to display
        p (float): Display segments with total attribution at least p
        percent (float): The fraction of top segments to display
        treshold (float): The treshold above which to display segments
        perturber (callable): Used to perturb lower attributed segments
    '''
    def __init__(
        self, k=None, p=None, percent=None, treshold=None, perturber=None
    ):
        self.perturber = perturber
        if perturber is None:
            self.perturber = SingleColorPerturber((0.5,0.5,0.5))
        if (k is None)+(p is None)+(percent is None)+(treshold is None) != 3:
            raise ValueError(
                'Exactly one of k, p, percent, and treshold should be set'
            )
        self.k = k
        self.p = p
        self.percent = percent
        self.treshold = treshold
        if not k is None:
            self.mode = 'k'
        elif not p is None:
            self.mode = 'p'
        elif not percent is None:
            self.mode = 'percent'
        else:
            self.mode = 'treshold'

    def __call__(self, vals, image, masks=None):
        '''
        Args:
            values (array): [S] array of attribution scores per segment
            image (array): Image to visualize attribution for of shape [H,W,C]
            masks (array): [S,H,W] array of 0/1 masks for the segments
        Returns:
            array: [H,W,C] image visualizing the attribution on the image
        '''
        if len(vals.shape) == 3:
            vals = np.squeeze(vals, axis=0)
        shape = vals.shape
        vals = vals.reshape(-1)
        to_perturb = np.ones((1,vals.size))
        if self.mode in ['percent', 'k']:
            if self.mode == 'percent':
                self.k = int(np.ceil(self.percent*vals.size))
            top_order = vals.argsort()
            to_perturb[0, top_order[:-self.k]] = 0
        elif self.mode == 'treshold':
            to_perturb[0, vals>self.treshold] = 0
        else:
            vals = vals-np.min(vals)
            vals = vals/np.sum(vals)
            top_order = vals.argsort()
            top_values = vals[top_order]
            total_value = 0
            for i, val in enumerate(top_values[::-1]):
                total_value+=val
                if total_value > self.p:
                    break
            to_perturb[0, top_order[:-(i+1)]] = 0
        if masks is None:
            distortion_mask = to_perturb.reshape(*shape)
        else:
            distortion_mask = perturbation_masks(masks, to_perturb)
        perturbed_image = self.perturber(
            image, distortion_mask, to_perturb, masks
        )[0]
        return perturbed_image[0]

class HeatmapVisualizer():
    '''
    Creates a numpy images showing the attribution of the image by creating a
    heatmap overlaid on top of the original image.

    Args:
        normalize (bool): Whether to normalize attributions in range [0,1]
        image_weight (float): How visible should the image be under the heatmap
        colormap (cv2.ColormapTypes): The OpenCV colormap that the heatmap uses
        invert_colormap (bool): Whether to swap the direction of the colormap
        color_by_rank (bool): If True, intensity corresponds to attribution rank
    '''
    def __init__(
            self, normalize=False, image_weight=0.5, colormap=None,
            invert_colormap=False, color_by_rank=False
        ):
        self.perturber = ReplaceImagePerturber(replace_images=None)
        self.normalize = normalize
        self.image_weight = image_weight
        import cv2
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        self.colormap = colormap
        self.invert_colormap = invert_colormap
        self.applyColorMap = cv2.applyColorMap
        self.color_by_rank = color_by_rank

    def __call__(self, vals, image, masks=None):
        '''
        Args:
            values (array): Array of attribution scores
            image (array): Image to visualize attribution for of shape [H,W,C]
            masks (array): [S,H,W] array of 0/1 masks for the segments
        Returns:
            array: [H,W,C] image visualizing the attribution on the image
        '''
        if masks is None:
            heatmap = vals
        else:
            heatmap = perturbation_masks(masks, vals.reshape((1,-1)))
        if self.color_by_rank:
            heatmap_shape = heatmap.shape
            heatmap = heatmap.reshape(-1)
            sidx = np.argsort(heatmap)
            idx = np.concatenate(
                ([0],np.flatnonzero(np.diff(heatmap[sidx]))+1,[heatmap.size])
            )
            heatmap = np.repeat(idx[:-1],np.diff(idx))[sidx.argsort()]
            heatmap = heatmap/np.max(heatmap)
            heatmap = heatmap.reshape(heatmap_shape)
        if len(heatmap.shape) == 3:
            heatmap = heatmap[0]
        if self.normalize:
            heatmap = heatmap-np.min(heatmap)
            heatmap = heatmap/np.max(heatmap)
        if self.invert_colormap:
            heatmap = 1-heatmap
        heatmap = (heatmap*255).astype(np.uint8)
        heatmap = self.applyColorMap(heatmap, colormap=self.colormap)
        perturbed_image = self.perturber(
            image, np.full(image.shape[:1], self.image_weight), None, masks,
            replace_images=(heatmap/255).astype(np.float32)
        )[0]
        return perturbed_image[0]

class AUCVisualizer():
    '''
    Makes a matplotlib plot showcasing the AUC scores for provided curves.
    Args:
        show (bool): Whether to directly show() the plot
    '''
    def __init__(self, show=True):
        self.show = show
    def __call__(self, *curves):
        '''
        Args:
            *curves: Curves to display of format values or (values, title str),
                where values is either a lif/mif curve or a tuple of (lif, mif)
        '''
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = plt.figure()
        titles = []
        handles = []
        for i, curve in enumerate(curves):
            if isinstance(curve[-1], str):
                titles.append(curve[-1])
                curve = curve[0]
            color = colors[i%len(colors)]
            if len(curve) == 2:
                lif, mif = curve
                line, = plt.plot(np.linspace(0,1,len(lif)), lif, c=color)
                plt.plot(np.linspace(0,1,len(mif)), mif, c=color)
            else:
                line, = plt.plot(np.linspace(0,1,len(curve)), curve, c=color)
            handles.append(line)
        if len(titles) > 0:
            plt.legend(handles, titles)
        if self.show:
            plt.show(block=False)
        return fig