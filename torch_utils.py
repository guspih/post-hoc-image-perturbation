import numpy as np
import torch
import torch.nn as nn

#from image_perturbers import cast_image


class TorchModelWrapper(nn.Module):
    def __init__(
        self, model, input_transforms, output_idxs=..., contrast=None,
        gpu=False
    ):
        super().__init__()
        self.model = model
        self.input_transform = input_transforms
        self.output_idxs = output_idxs
        self.contrast = contrast
        self.gpu = gpu

        if gpu:
            model.cuda()
    
    def __call__(self, x):
        x = self.input_transform(x)
        if self.gpu:
            x = x.cuda()
        y = self.model(x).numpy(force=True)[:, self.output_idxs]
        if not self.contrast is None:
            positive = np.sum(y[:,self.contrast[0]], axis=1)
            y = positive - np.sum(y[:,self.contrast[1]], axis=1)
        return y
    
class ImageToTorch(nn.Module):
    def __call__(self, pic):
        tensor = torch.from_numpy(pic.astype(np.float32))
        if len(tensor.shape) == 3:
            tensor = torch.unsqueeze(tensor, dim=0)
        return tensor.permute((0,3,1,2))